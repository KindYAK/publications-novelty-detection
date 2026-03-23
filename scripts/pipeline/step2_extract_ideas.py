"""Step 2: Chunk full texts and extract ideas via async LLM calls."""
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

# Force line-buffered output for progress visibility
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import pandas as pd

from config import (
    CHUNK_MAX_TOKENS,
    EXP_DATA,
    EXP_REPORTS,
    EXTRACTION_CONCURRENCY,
    LLM_MODEL,
    LLM_REASONING,
    MAX_RETRIES,
    MIN_CHUNK_TOKENS,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)

# Use char-based approximation for speed (4 chars ≈ 1 token for English)
CHARS_PER_TOKEN = 4
MAX_CHUNK_CHARS = CHUNK_MAX_TOKENS * CHARS_PER_TOKEN  # ~12000 chars
MIN_CHUNK_CHARS = MIN_CHUNK_TOKENS * CHARS_PER_TOKEN  # ~320 chars
OVERLAP_CHARS = 800  # ~200 tokens overlap


def chunk_text(text: str) -> list[str]:
    """Split text into large chunks by paragraphs/sentences, char-based sizing."""
    text = text.replace("\r\n", "\n").strip()
    if len(text) < MIN_CHUNK_CHARS:
        return []

    # Split into paragraphs first
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        if current_len + para_len > MAX_CHUNK_CHARS and current:
            chunk = "\n\n".join(current)
            if len(chunk) >= MIN_CHUNK_CHARS:
                chunks.append(chunk)
            # Overlap: keep last paragraph
            current = [current[-1]] if current else []
            current_len = len(current[0]) if current else 0
        current.append(para)
        current_len += para_len

    if current:
        chunk = "\n\n".join(current)
        if len(chunk) >= MIN_CHUNK_CHARS:
            chunks.append(chunk)

    return chunks


# === LLM extraction ===
_model_to_use = LLM_MODEL


async def extract_chunk(client, semaphore, title: str, chunk: str) -> tuple[list[str], int, int]:
    """Extract ideas from one chunk. Returns (ideas, tokens_in, tokens_out)."""
    global _model_to_use

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(title=title, chunk=chunk[:12000])},
    ]

    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                resp = await client.chat.completions.create(
                    model=_model_to_use,
                    messages=messages,
                    response_format={"type": "json_object"},
                    max_completion_tokens=300,
                    reasoning_effort=LLM_REASONING,
                )

            content = resp.choices[0].message.content
            data = json.loads(content)
            ideas = data.get("ideas", [])
            ideas = [i.strip() for i in ideas if isinstance(i, str) and 3 < len(i.strip()) < 300]
            t_in = resp.usage.prompt_tokens if resp.usage else 0
            t_out = resp.usage.completion_tokens if resp.usage else 0
            return ideas, t_in, t_out

        except Exception as e:
            err_str = str(e).lower()
            # Rate limit
            if "rate" in err_str or "429" in err_str:
                wait = min(2**attempt * 5, 60)
                await asyncio.sleep(wait)
                continue
            # Other errors
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2)
                continue
            return [], 0, 0
    return [], 0, 0


async def run_extraction(subset: pd.DataFrame, test_n: int | None = None):
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(EXTRACTION_CONCURRENCY)

    if test_n:
        subset = subset.head(test_n)

    # Build tasks
    tasks = []
    papers_processed = 0
    papers_no_text = 0
    for _, row in subset.iterrows():
        ft_path = row.get("fulltext_path")
        if not ft_path or not Path(ft_path).exists():
            papers_no_text += 1
            continue
        text = Path(ft_path).read_text(encoding="utf-8", errors="replace")
        if len(text.strip()) < 200:
            papers_no_text += 1
            continue
        chunks = chunk_text(text)
        if not chunks:
            papers_no_text += 1
            continue
        papers_processed += 1
        title = str(row.get("title", ""))
        acl_id = str(row["acl_id"])
        for ci, chunk in enumerate(chunks):
            tasks.append((acl_id, title, ci, chunk))

    print(f"Papers with text: {papers_processed}, skipped: {papers_no_text}")
    print(f"Total chunks to process: {len(tasks)}")

    # Run async extraction
    results = []
    total_in = 0
    total_out = 0
    errors = 0
    t0 = time.time()

    async def process(acl_id, title, ci, chunk):
        nonlocal total_in, total_out, errors
        ideas, t_in, t_out = await extract_chunk(client, semaphore, title, chunk)
        total_in += t_in
        total_out += t_out
        if not ideas:
            errors += 1
        return {"acl_id": acl_id, "chunk_idx": ci, "ideas": ideas}

    coros = [process(*t) for t in tasks]

    # Progress tracking
    done = 0
    for batch_start in range(0, len(coros), 500):
        batch = coros[batch_start : batch_start + 500]
        batch_results = await asyncio.gather(*batch, return_exceptions=True)
        for r in batch_results:
            done += 1
            if isinstance(r, Exception):
                errors += 1
                results.append({"acl_id": "error", "chunk_idx": -1, "ideas": []})
            else:
                results.append(r)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        est_cost_in = total_in / 1_000_000 * 0.15  # rough estimate
        est_cost_out = total_out / 1_000_000 * 0.60
        print(
            f"  [{done}/{len(coros)}] {rate:.0f}/s | "
            f"tokens: {total_in:,}in {total_out:,}out | "
            f"~${est_cost_in + est_cost_out:.2f} | "
            f"errors: {errors} | model: {_model_to_use}"
        )

    elapsed = time.time() - t0
    return results, {
        "total_chunks": len(tasks),
        "total_papers": papers_processed,
        "total_tokens_in": total_in,
        "total_tokens_out": total_out,
        "errors": errors,
        "elapsed_s": elapsed,
        "model_used": _model_to_use,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, default=None, help="Process only N papers for testing")
    args = parser.parse_args()

    subset = pd.read_parquet(EXP_DATA / "subset.parquet")
    print(f"Loaded subset: {len(subset)} papers")

    results, stats = asyncio.run(run_extraction(subset, test_n=args.test))

    # Save ideas
    out_path = EXP_DATA / "ideas.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(results)} idea records to {out_path}")

    # Stats
    all_ideas = [idea for r in results for idea in r["ideas"]]
    papers_with_ideas = len(set(r["acl_id"] for r in results if r["ideas"]))
    print(f"Total ideas extracted: {len(all_ideas)}")
    print(f"Unique ideas: {len(set(all_ideas))}")
    print(f"Papers with >=1 idea: {papers_with_ideas}")
    print(f"Avg ideas per chunk: {len(all_ideas) / max(len(results), 1):.1f}")
    print(f"Model used: {stats['model_used']}")
    print(f"Total tokens: {stats['total_tokens_in']:,} in, {stats['total_tokens_out']:,} out")
    print(f"Elapsed: {stats['elapsed_s']:.0f}s")

    # Write report
    report = f"""# Idea Extraction Report

## Config
- Model: {stats['model_used']}
- Concurrency: {EXTRACTION_CONCURRENCY}
- Chunk size: {CHUNK_MAX_TOKENS} tokens
- Test mode: {args.test or 'full'}

## Results
- Papers processed: {stats['total_papers']:,}
- Chunks processed: {stats['total_chunks']:,}
- Total ideas: {len(all_ideas):,}
- Unique ideas: {len(set(all_ideas)):,}
- Papers with ideas: {papers_with_ideas:,}
- Avg ideas/chunk: {len(all_ideas) / max(len(results), 1):.1f}
- Errors: {stats['errors']}

## Cost
- Tokens in: {stats['total_tokens_in']:,}
- Tokens out: {stats['total_tokens_out']:,}
- Elapsed: {stats['elapsed_s']:.0f}s

## Sample Ideas
"""
    for idea in all_ideas[:20]:
        report += f"- {idea}\n"
    (EXP_REPORTS / "02_extraction_report.md").write_text(report, encoding="utf-8")
    print("Report: reports/02_extraction_report.md")


if __name__ == "__main__":
    main()
