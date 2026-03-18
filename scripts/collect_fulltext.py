"""Full-text collection for arXiv papers — multi-strategy, async, resume-safe.

Strategies (in priority order):
  1. ar5iv HTML (arxiv's HTML renders) — fast, clean text
  2. arXiv LaTeX source (e-print) — highest quality, more complex
  3. PDF via PyMuPDF — universal fallback

Rate limits:
  - arXiv: ~4 req/s across all endpoints
  - ar5iv: separate service, more lenient but still be polite
  - We use async with semaphore to control concurrency
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import tarfile
import time
from io import BytesIO
from pathlib import Path

import aiohttp
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

DATA_DIR = Path("data/processed/v3")
FULLTEXT_DIR = DATA_DIR / "fulltext"
PROGRESS_FILE = DATA_DIR / "fulltext_progress.json"

# Concurrency / rate limiting
CONCURRENT_REQUESTS = 16  # simultaneous downloads
REQUEST_DELAY = 0.05  # seconds between requests (per worker)
SAVE_INTERVAL = 1000  # save progress every N papers

# Timeouts
TIMEOUT = aiohttp.ClientTimeout(total=30, connect=10)
HEADERS = {"User-Agent": "novelty-search/0.1 (academic research; polite bot)"}

# ar5iv HTML endpoint
AR5IV_URL = "https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
# arXiv source endpoint
EPRINT_URL = "https://arxiv.org/e-print/{arxiv_id}"
# arXiv PDF endpoint
PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"


# ═══════════════════════════════════════════════════════════════
# TEXT EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_text_from_html(html: str) -> str | None:
    """Extract clean text from ar5iv HTML page."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    # Remove script, style, nav elements
    for tag in soup.find_all(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    # Remove figures, tables (keep captions)
    for fig in soup.find_all("figure"):
        caption = fig.find("figcaption")
        if caption:
            fig.replace_with(caption)
        else:
            fig.decompose()

    # Get the main article content
    article = soup.find("article") or soup.find("div", class_="ltx_page_content")
    if article:
        text = article.get_text(separator="\n", strip=True)
    else:
        text = soup.get_text(separator="\n", strip=True)

    # Clean up
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()

    # Sanity check — ar5iv error pages are short
    if len(text) < 500:
        return None

    return text


def extract_text_from_latex_bytes(data: bytes) -> str | None:
    """Extract text from LaTeX source (tar.gz or single .tex file)."""
    tex_content = None

    # Try as tar.gz first
    try:
        with tarfile.open(fileobj=BytesIO(data), mode="r:gz") as tar:
            tex_files = [m for m in tar.getmembers() if m.name.endswith(".tex") and m.isfile()]
            if not tex_files:
                return None

            # Prefer main.tex, paper.tex, etc.
            main_candidates = [f for f in tex_files if Path(f.name).stem in ("main", "paper", "article", "ms")]
            if main_candidates:
                chosen = main_candidates[0]
            else:
                chosen = max(tex_files, key=lambda f: f.size)

            f = tar.extractfile(chosen)
            if f:
                tex_content = f.read().decode("utf-8", errors="replace")
    except (tarfile.TarError, EOFError):
        # Maybe it's a single .tex file (gzipped)
        import gzip
        try:
            tex_content = gzip.decompress(data).decode("utf-8", errors="replace")
        except Exception:
            # Maybe raw .tex
            try:
                tex_content = data.decode("utf-8", errors="replace")
                if "\\begin{document}" not in tex_content:
                    return None
            except Exception:
                return None

    if not tex_content:
        return None

    return strip_latex(tex_content)


def strip_latex(text: str) -> str:
    """LaTeX to plaintext — fast, good enough for LLM processing."""
    # Remove comments
    text = re.sub(r"(?<!\\)%.*$", "", text, flags=re.MULTILINE)

    # Extract document body
    doc_match = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", text, re.DOTALL)
    if doc_match:
        text = doc_match.group(1)

    # Remove environments we don't need
    for env in ("figure", "table", "tikzpicture", "algorithm", "lstlisting", "verbatim"):
        text = re.sub(rf"\\begin\{{{env}\*?\}}.*?\\end\{{{env}\*?\}}", "", text, flags=re.DOTALL)

    # Replace math with [formula]
    text = re.sub(r"\\\[.*?\\\]", " [formula] ", text, flags=re.DOTALL)
    text = re.sub(r"\\begin\{equation\*?\}.*?\\end\{equation\*?\}", " [formula] ", text, flags=re.DOTALL)
    text = re.sub(r"\\begin\{align\*?\}.*?\\end\{align\*?\}", " [formula] ", text, flags=re.DOTALL)
    text = re.sub(r"\$\$.*?\$\$", " [formula] ", text, flags=re.DOTALL)

    # Section headers
    text = re.sub(r"\\(?:section|subsection|subsubsection|paragraph)\*?\{([^}]*)\}", r"\n\n\1\n\n", text)
    # Formatting commands
    text = re.sub(r"\\(?:textbf|textit|emph|underline|texttt)\{([^}]*)\}", r"\1", text)
    # References
    text = re.sub(r"\\(?:cite|citep|citet|ref|label|eqref|autoref)\{[^}]*\}", "", text)
    # Remove remaining commands
    text = re.sub(r"\\(?:begin|end)\{[^}]*\}", "", text)
    text = re.sub(r"\\item\b", "- ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+\*?", "", text)

    # Clean up braces, whitespace
    text = re.sub(r"[{}]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    text = text.strip()
    return text if len(text) > 300 else None


def extract_text_from_pdf_bytes(data: bytes) -> str | None:
    """Extract text from PDF using PyMuPDF."""
    import fitz

    try:
        doc = fitz.open(stream=data, filetype="pdf")
        parts = []
        for page in doc:
            parts.append(page.get_text())
        doc.close()
        text = "\n".join(parts)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = text.strip()
        return text if len(text) > 300 else None
    except Exception as e:
        logger.debug("PDF extraction failed: %s", e)
        return None


# ═══════════════════════════════════════════════════════════════
# ASYNC DOWNLOAD PIPELINE
# ═══════════════════════════════════════════════════════════════

async def try_ar5iv(session: aiohttp.ClientSession, arxiv_id: str) -> tuple[str | None, str]:
    """Try to get text from ar5iv HTML render."""
    url = AR5IV_URL.format(arxiv_id=arxiv_id)
    html_timeout = aiohttp.ClientTimeout(total=20, connect=8)
    try:
        async with session.get(url, timeout=html_timeout) as resp:
            if resp.status == 200:
                html = await resp.text()
                text = extract_text_from_html(html)
                if text:
                    return text, "html"
            elif resp.status != 404:
                logger.debug("ar5iv %s: HTTP %d", arxiv_id, resp.status)
    except Exception as e:
        logger.debug("ar5iv %s: %s", arxiv_id, e)
    return None, ""


async def try_latex(session: aiohttp.ClientSession, arxiv_id: str) -> tuple[str | None, str]:
    """Try to get text from LaTeX source."""
    url = EPRINT_URL.format(arxiv_id=arxiv_id)
    try:
        async with session.get(url) as resp:
            if resp.status == 200:
                data = await resp.read()
                ct = resp.headers.get("Content-Type", "")
                if "pdf" in ct.lower():
                    # Source is PDF-only, skip (we'll handle PDF separately)
                    return None, ""
                text = extract_text_from_latex_bytes(data)
                if text:
                    return text, "latex"
            elif resp.status != 404:
                logger.debug("e-print %s: HTTP %d", arxiv_id, resp.status)
    except Exception as e:
        logger.debug("e-print %s: %s", arxiv_id, e)
    return None, ""


async def try_pdf(session: aiohttp.ClientSession, arxiv_id: str) -> tuple[str | None, str]:
    """Try to get text from PDF."""
    url = PDF_URL.format(arxiv_id=arxiv_id)
    try:
        async with session.get(url) as resp:
            if resp.status == 200:
                data = await resp.read()
                text = extract_text_from_pdf_bytes(data)
                if text:
                    return text, "pdf"
            elif resp.status != 404:
                logger.debug("pdf %s: HTTP %d", arxiv_id, resp.status)
    except Exception as e:
        logger.debug("pdf %s: %s", arxiv_id, e)
    return None, ""


async def fetch_fulltext(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    arxiv_id: str,
    strategies: list[str],
) -> tuple[str, str | None, str]:
    """Fetch full text using multiple strategies. Returns (arxiv_id, text, source)."""
    async with semaphore:
        for strategy in strategies:
            if strategy == "html":
                text, src = await try_ar5iv(session, arxiv_id)
            elif strategy == "latex":
                text, src = await try_latex(session, arxiv_id)
            elif strategy == "pdf":
                text, src = await try_pdf(session, arxiv_id)
            else:
                continue

            if text:
                return arxiv_id, text, src

            await asyncio.sleep(REQUEST_DELAY)

        return arxiv_id, None, "failed"


# ═══════════════════════════════════════════════════════════════
# PROGRESS / STORAGE
# ═══════════════════════════════════════════════════════════════

def load_progress() -> dict:
    """Load progress tracking file."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"completed": {}, "stats": {"html": 0, "latex": 0, "pdf": 0, "failed": 0}}


def save_progress(progress: dict):
    """Save progress atomically."""
    tmp = PROGRESS_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(progress, f)
    tmp.replace(PROGRESS_FILE)


def save_text(arxiv_id: str, text: str, source: str):
    """Save extracted text to a file."""
    # Use flat structure with safe filename
    safe_id = arxiv_id.replace("/", "_").replace(".", "_")
    out_file = FULLTEXT_DIR / f"{safe_id}.txt"
    out_file.write_text(text, encoding="utf-8")


# ═══════════════════════════════════════════════════════════════
# BATCH PROCESSING
# ═══════════════════════════════════════════════════════════════

async def process_batch(
    arxiv_ids: list[str],
    strategies: list[str],
    progress: dict,
    batch_label: str = "",
) -> dict:
    """Process a batch of papers. Returns updated progress."""
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS * 2, ttl_dns_cache=300)
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=TIMEOUT,
        headers=HEADERS,
    ) as session:
        # Create tasks
        tasks = []
        for aid in arxiv_ids:
            if aid in progress["completed"]:
                continue
            tasks.append(fetch_fulltext(session, semaphore, aid, strategies))

        if not tasks:
            logger.info("All papers in batch already processed")
            return progress

        logger.info("Processing %d papers %s", len(tasks), batch_label)

        # Process with progress bar
        completed = 0
        pbar = tqdm(total=len(tasks), desc=f"Fulltext {batch_label}", unit="paper")

        for coro in asyncio.as_completed(tasks):
            arxiv_id, text, source = await coro
            completed += 1

            if text:
                save_text(arxiv_id, text, source)
                progress["completed"][arxiv_id] = source
                progress["stats"][source] = progress["stats"].get(source, 0) + 1
            else:
                progress["completed"][arxiv_id] = "failed"
                progress["stats"]["failed"] = progress["stats"].get("failed", 0) + 1

            pbar.update(1)

            if completed % SAVE_INTERVAL == 0:
                save_progress(progress)
                stats = progress["stats"]
                pbar.set_postfix(
                    html=stats.get("html", 0),
                    latex=stats.get("latex", 0),
                    pdf=stats.get("pdf", 0),
                    fail=stats.get("failed", 0),
                )

        pbar.close()
        save_progress(progress)

    return progress


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

async def main():
    FULLTEXT_DIR.mkdir(parents=True, exist_ok=True)

    # Load paper IDs
    parquet = DATA_DIR / "arxiv_ids_all.parquet"
    df = pd.read_parquet(parquet, columns=["arxiv_id"])
    all_ids = df["arxiv_id"].tolist()
    logger.info("Total papers: %d", len(all_ids))

    # Load progress
    progress = load_progress()
    done = len(progress["completed"])
    logger.info("Already completed: %d", done)
    remaining = [aid for aid in all_ids if aid not in progress["completed"]]
    logger.info("Remaining: %d", len(remaining))

    if not remaining:
        logger.info("All papers processed!")
        print_stats(progress)
        return

    # Strategies: try HTML first (fastest), then LaTeX, then PDF
    # HTML first (fast, 97% coverage), PDF fallback (fast, good quality)
    # Skip LaTeX: slower downloads + produces less text than PDF
    strategies = ["html", "pdf"]

    # Process in batches of 5000 for progress reporting
    batch_size = 5000
    for i in range(0, len(remaining), batch_size):
        batch = remaining[i:i + batch_size]
        label = f"[{i+1}-{min(i+batch_size, len(remaining))}/{len(remaining)}]"
        progress = await process_batch(batch, strategies, progress, label)
        print_stats(progress)

    logger.info("DONE!")
    print_stats(progress)


def print_stats(progress: dict):
    stats = progress["stats"]
    total = sum(stats.values())
    success = total - stats.get("failed", 0)
    print(f"\n{'='*50}")
    print(f"Full-text collection stats")
    print(f"{'='*50}")
    print(f"Total processed: {total:,}")
    print(f"Success:         {success:,} ({success/max(total,1)*100:.1f}%)")
    for src in ["html", "latex", "pdf", "failed"]:
        n = stats.get(src, 0)
        print(f"  {src:>8}: {n:>8,} ({n/max(total,1)*100:.1f}%)")
    print()


if __name__ == "__main__":
    asyncio.run(main())
