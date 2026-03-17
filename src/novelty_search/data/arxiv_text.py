"""arXiv full-text downloader — fetches LaTeX source or plain text from arXiv.

For papers that have arXiv IDs, we can get clean text (no PDF parsing needed).
"""

from __future__ import annotations

import logging
import tarfile
import time
from io import BytesIO
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

ARXIV_API_URL = "http://export.arxiv.org/api/query"
ARXIV_SOURCE_URL = "https://arxiv.org/e-print/{arxiv_id}"


def fetch_abstract_via_api(arxiv_id: str) -> str | None:
    """Fetch a paper's abstract from the arXiv API.

    This is a fallback — Semantic Scholar already gives us abstracts.
    """
    params = {"id_list": arxiv_id, "max_results": 1}
    resp = requests.get(ARXIV_API_URL, params=params, timeout=30)
    resp.raise_for_status()

    # Simple XML parsing for the abstract
    text = resp.text
    start = text.find("<summary>")
    end = text.find("</summary>")
    if start == -1 or end == -1:
        return None
    return text[start + len("<summary>") : end].strip()


def download_latex_source(arxiv_id: str, output_dir: Path) -> Path | None:
    """Download and extract LaTeX source files for a paper.

    Args:
        arxiv_id: e.g. "1706.03762"
        output_dir: Directory to extract files into.

    Returns:
        Path to the extracted directory, or None if download failed.
    """
    url = ARXIV_SOURCE_URL.format(arxiv_id=arxiv_id)
    time.sleep(0.5)  # respect arXiv rate limit (1 req/sec)

    try:
        resp = requests.get(url, timeout=60, headers={"User-Agent": "novelty-search/0.1"})
        if resp.status_code == 404:
            logger.warning("No source available for %s", arxiv_id)
            return None
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning("Failed to download %s: %s", arxiv_id, e)
        return None

    paper_dir = output_dir / arxiv_id.replace("/", "_")
    paper_dir.mkdir(parents=True, exist_ok=True)

    content_type = resp.headers.get("Content-Type", "")

    if "application/x-eprint-tar" in content_type or "application/gzip" in content_type:
        try:
            with tarfile.open(fileobj=BytesIO(resp.content), mode="r:gz") as tar:
                tar.extractall(paper_dir, filter="data")
            return paper_dir
        except (tarfile.TarError, Exception) as e:
            logger.warning("Failed to extract tar for %s: %s", arxiv_id, e)
    elif "application/pdf" in content_type:
        # Some papers only have PDF, no LaTeX source
        pdf_path = paper_dir / "paper.pdf"
        pdf_path.write_bytes(resp.content)
        logger.info("Only PDF available for %s", arxiv_id)
        return None
    else:
        # Might be a single .tex file (not tarred)
        tex_path = paper_dir / "main.tex"
        tex_path.write_bytes(resp.content)
        return paper_dir

    return None


def extract_text_from_latex(source_dir: Path) -> str | None:
    """Extract plain text from LaTeX source files.

    Simple approach: find .tex files, strip LaTeX commands, return concatenated text.
    """
    tex_files = list(source_dir.glob("*.tex"))
    if not tex_files:
        return None

    # Prefer main.tex, otherwise take the largest .tex file
    main_candidates = [f for f in tex_files if f.stem in ("main", "paper", "article")]
    if main_candidates:
        tex_file = main_candidates[0]
    else:
        tex_file = max(tex_files, key=lambda f: f.stat().st_size)

    try:
        content = tex_file.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    return strip_latex(content)


def strip_latex(text: str) -> str:
    """Rough LaTeX-to-plaintext conversion.

    Not perfect, but good enough for embedding and LLM processing.
    """
    import re

    # Remove comments
    text = re.sub(r"(?<!\\)%.*$", "", text, flags=re.MULTILINE)

    # Extract content between \begin{document} and \end{document}
    doc_match = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", text, re.DOTALL)
    if doc_match:
        text = doc_match.group(1)

    # Remove common environments we don't need
    for env in ("figure", "table", "tikzpicture", "algorithm", "lstlisting"):
        text = re.sub(rf"\\begin\{{{env}\}}.*?\\end\{{{env}\}}", "", text, flags=re.DOTALL)

    # Remove math display environments but keep inline math readable
    text = re.sub(r"\\\[.*?\\\]", " [formula] ", text, flags=re.DOTALL)
    text = re.sub(r"\\begin\{equation\*?\}.*?\\end\{equation\*?\}", " [formula] ", text, flags=re.DOTALL)
    text = re.sub(r"\\begin\{align\*?\}.*?\\end\{align\*?\}", " [formula] ", text, flags=re.DOTALL)
    text = re.sub(r"\$\$.*?\$\$", " [formula] ", text, flags=re.DOTALL)

    # Keep inline math as-is (LLM can handle simple $x$)
    # Remove common commands
    text = re.sub(r"\\(?:section|subsection|subsubsection|paragraph)\*?\{([^}]*)\}", r"\n\n\1\n\n", text)
    text = re.sub(r"\\(?:textbf|textit|emph|underline)\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:cite|ref|label|eqref)\{[^}]*\}", "", text)
    text = re.sub(r"\\(?:begin|end)\{[^}]*\}", "", text)
    text = re.sub(r"\\item\b", "- ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+\*?", "", text)

    # Clean up
    text = re.sub(r"\{|\}", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()
