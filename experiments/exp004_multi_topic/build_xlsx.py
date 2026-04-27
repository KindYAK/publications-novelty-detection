#!/usr/bin/env python3
"""Build human-readable XLSX files showing the novelty detection pipeline.

Files produced:
  01_cross_topic_summary.xlsx       - Cross-topic results (1 sheet)
  embeddings_01_articles.xlsx       - 5234 articles with metadata + novelty scores
  embeddings_02_chunks_to_ideas.xlsx - Live chunk -> idea extraction examples (~150 papers)
  embeddings_03_base_ideas.xlsx     - The 100 base idea clusters + sample papers
  embeddings_04_paper_idea_weights.xlsx - Per-paper idea spectrum (top ideas with weights)
  embeddings_05_metrics_explained.xlsx - All metrics + golden set + ranking
"""
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent.parent
EXP1 = PROJECT / "experiments" / "exp001_embeddings" / "data"
EXP4 = PROJECT / "experiments" / "exp004_multi_topic"
OUT = EXP4 / "xlsx_outputs"
OUT.mkdir(parents=True, exist_ok=True)

DATA_V3 = PROJECT / "data" / "processed" / "v3"
ACL_FT = DATA_V3 / "acl_all_fulltext"

# Load .env
_env = PROJECT / ".env"
if _env.exists():
    for line in _env.read_text(encoding="utf-8").strip().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


# ============================================================================
# FILE 1: CROSS-TOPIC SUMMARY
# ============================================================================
def build_cross_topic_summary():
    print("Building 01_cross_topic_summary.xlsx ...")
    with open(EXP4 / "all_results.json") as f:
        all_results = json.load(f)

    rows = []
    for r in all_results:
        if "error" in r:
            continue
        m = r.get("metrics", {})
        td = r.get("trash_detection", {})
        # Build top golden paper info
        gh = r.get("golden_high_results", [])
        top_golden = ""
        if gh:
            best = max(gh, key=lambda g: g.get("knn_pct") or 0)
            top_golden = f"{best['query'][:30]} -> {best.get('knn_pct', 0):.0f}%"

        rows.append({
            "Topic": r["topic"].replace("_", " ").title(),
            "Papers analyzed": r["n_papers"],
            "Test set size (>=split year)": r["n_test"],
            "Split year": r["split_year"],
            "kNN max-pool (Spearman r)": round(m.get("knn_max_pool", {}).get("sp_r", float("nan")), 4),
            "kNN mean-pool (Spearman r)": round(m.get("knn_mean_pool", {}).get("sp_r", float("nan")), 4),
            "Mahalanobis D_M (Spearman r)": round(m.get("D_M", {}).get("sp_r", float("nan")), 4),
            "Rarity R (Spearman r)": round(m.get("R", {}).get("sp_r", float("nan")), 4),
            "Coherence C (Spearman r)": round(m.get("C", {}).get("sp_r", float("nan")), 4),
            "BPI original (Spearman r)": round(m.get("BPI_original", {}).get("sp_r", float("nan")), 4),
            "Combined formula (Spearman r)": round(m.get("combined_best", {}).get("sp_r", float("nan")), 4),
            "Trash detect: # flagged": td.get("n_flagged", 0),
            "Trash detect: precision": f"{td.get('precision_le19', 0)*100:.1f}%" if td else "—",
            "Top golden paper rank": top_golden,
        })

    df = pd.DataFrame(rows)

    # Add MEAN row (excluding overall_mixed)
    topic_rows = df[df["Topic"] != "Overall Mixed"]
    mean_row = {"Topic": "*** MEAN (10 topics) ***"}
    for col in df.columns[1:]:
        if col == "Topic": continue
        try:
            vals = pd.to_numeric(topic_rows[col], errors="coerce").dropna()
            if len(vals): mean_row[col] = round(vals.mean(), 4)
        except Exception:
            pass
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    explanation = pd.DataFrame({
        "What this file shows": [
            "Results of running the novelty detection pipeline on 11 topics (10 specific + 1 cross-topic mixed).",
            "",
            "FOR EACH TOPIC: papers were filtered by keywords from 442K arXiv + 71K ACL papers.",
            "Each paper went through: full text -> chunks -> LLM idea extraction -> embeddings -> metrics.",
            "",
            "WHAT THE METRICS MEAN:",
            "• kNN max-pool: 'How different is this paper from its 10 nearest neighbors in idea space?' Higher = more novel.",
            "• kNN mean-pool: same but using average of all paper's ideas instead of max-pool.",
            "• Mahalanobis D_M: 'How far from the corpus mean idea profile?' (accounts for correlations).",
            "• Rarity R: 'How rare are this paper's ideas relative to corpus prevalence?' (-Sum s_i log p_i)",
            "• Coherence C: 'How focused is this paper on few ideas?' (Sum of squared idea weights)",
            "• BPI original: R * C * (1/density) — the formula proposed in the paper.",
            "• Combined: kNN^0.5 * D_M^0.5 * R^0.1 — the best ensemble from exp002.",
            "",
            "Spearman r is the rank correlation with log(citations+1). r near 0 = no signal, r > 0.2 = decent signal.",
            "",
            "KEY FINDINGS:",
            "1) BPI_original works across ALL 10 topics (mean r=+0.22), opposite of the original embeddings finding.",
            "2) kNN is weaker here (mean r=+0.06) due to citation-biased sampling.",
            "3) Cross-topic comparison fails — novelty is topic-relative.",
            "4) Newer/niche topics show stronger signal (diffusion: r=+0.32).",
            "",
            "Total cost: $16.55 for extraction across all topics (gpt-5.4-nano).",
        ]
    })

    out_path = OUT / "01_cross_topic_summary.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        explanation.to_excel(w, sheet_name="What is this", index=False)
        df.to_excel(w, sheet_name="Results", index=False)
        # Auto-width
        for sheet in w.sheets.values():
            for col in sheet.columns:
                max_len = max((len(str(c.value)) for c in col if c.value), default=10)
                sheet.column_dimensions[col[0].column_letter].width = min(max_len + 2, 60)
    print(f"  saved: {out_path}")


# ============================================================================
# Helpers for embeddings topic
# ============================================================================
def find_fulltext(acl_id):
    if not isinstance(acl_id, str) or not acl_id:
        return None
    fname = acl_id.replace(".", "_").replace("/", "_") + ".txt"
    p = ACL_FT / fname
    return p if p.exists() else None


def chunk_text(text, max_chars=8000, min_chars=320):
    """Chunk by paragraphs, falling back to sentences if no paragraph breaks."""
    text = text.replace("\r\n", "\n").strip()
    if len(text) < min_chars:
        return []

    # Try paragraph splitting first
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # If single 'paragraph' (no breaks) is too long, split by sentences
    if len(paragraphs) == 1 and len(paragraphs[0]) > max_chars:
        # Split by sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", paragraphs[0])
        paragraphs = [s.strip() for s in sentences if s.strip()]

    chunks, current, current_len = [], [], 0
    for para in paragraphs:
        # If a single piece is still too long, split by char count
        if len(para) > max_chars:
            # Flush current
            if current:
                chunk = " ".join(current)
                if len(chunk) >= min_chars:
                    chunks.append(chunk)
                current, current_len = [], 0
            # Hard split
            for i in range(0, len(para), max_chars - 200):  # 200 char overlap
                piece = para[i:i + max_chars]
                if len(piece) >= min_chars:
                    chunks.append(piece)
            continue

        if current_len + len(para) > max_chars and current:
            chunk = " ".join(current)
            if len(chunk) >= min_chars:
                chunks.append(chunk)
            current = [current[-1]] if current else []
            current_len = len(current[0]) if current else 0
        current.append(para)
        current_len += len(para)
    if current:
        chunk = " ".join(current)
        if len(chunk) >= min_chars:
            chunks.append(chunk)
    return chunks


# ============================================================================
# Re-extract ideas for sample papers (for chunks_to_ideas examples)
# ============================================================================
SYSTEM_PROMPT = "You extract scientific ideas from paper excerpts. Output JSON only."
USER_PROMPT = """Extract novel ideas from this scientific paper excerpt.
- Return JSON: {{"ideas": ["idea1", "idea2", ...]}}
- 0-5 ideas, each 5-20 words, specific
- Focus on novel methods, techniques, models, findings
- Skip background, related work, references

Title: {title}

{chunk}"""


async def extract_examples(papers_df, max_chunks_per_paper=3):
    """Extract a few chunks per paper for showing examples."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(50)

    tasks = []
    for _, row in papers_df.iterrows():
        ft_path = find_fulltext(row["acl_id"])
        if not ft_path:
            continue
        text = ft_path.read_text(encoding="utf-8", errors="replace")
        chunks = chunk_text(text)
        n_chunks = min(max_chunks_per_paper, len(chunks))
        if n_chunks == 0:
            continue
        if n_chunks == 1:
            picked_chunks = [(0, chunks[0])]
        elif n_chunks == 2:
            picked_chunks = [(0, chunks[0]), (len(chunks) // 2, chunks[len(chunks) // 2])]
        else:
            # First + middle + last for variety (intro, methods, results)
            picked_chunks = [
                (0, chunks[0]),
                (len(chunks) // 2, chunks[len(chunks) // 2]),
                (len(chunks) - 1, chunks[-1]),
            ]
        for ci, ck in picked_chunks:
            tasks.append((row["acl_id"], row["title"], ci, ck, len(chunks)))

    print(f"    {len(tasks)} chunks to extract from {len(papers_df)} papers")

    async def process_one(acl_id, title, ci, chunk, total_chunks):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(title=title, chunk=chunk[:12000])},
        ]
        for attempt in range(3):
            try:
                async with semaphore:
                    resp = await client.chat.completions.create(
                        model="gpt-5.4-nano",
                        messages=messages,
                        response_format={"type": "json_object"},
                        max_completion_tokens=200,
                    )
                content = resp.choices[0].message.content
                data = json.loads(content)
                ideas = data.get("ideas", [])
                ideas = [i.strip() for i in ideas if isinstance(i, str) and 3 < len(i.strip()) < 300]
                return {"acl_id": acl_id, "title": title, "chunk_idx": ci, "total_chunks": total_chunks,
                        "chunk_text": chunk, "ideas": ideas}
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(1)
                else:
                    return {"acl_id": acl_id, "title": title, "chunk_idx": ci, "total_chunks": total_chunks,
                            "chunk_text": chunk, "ideas": [], "error": str(e)[:100]}

    results = []
    coros = [process_one(*t) for t in tasks]
    t0 = time.time()
    done = 0
    for batch_start in range(0, len(coros), 100):
        batch = coros[batch_start:batch_start + 100]
        batch_results = await asyncio.gather(*batch, return_exceptions=True)
        for r in batch_results:
            done += 1
            if isinstance(r, Exception):
                continue
            results.append(r)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        print(f"    [{done}/{len(coros)}] {rate:.0f}/s")
    return results


def select_sample_papers(metrics_df, n_top=40, n_bottom=40, n_middle=40, golden_titles=None):
    """Select interesting representative papers."""
    df = metrics_df.dropna(subset=["p_nn_novelty"]).copy()
    df = df.sort_values("p_nn_novelty", ascending=False)

    selected = []
    seen = set()

    # Golden set first
    if golden_titles:
        for t in golden_titles:
            mask = df["title"].str.contains(t, case=False, na=False, regex=False)
            if mask.any():
                pid = df[mask].iloc[0]["paper_id"]
                if pid not in seen:
                    selected.append(pid)
                    seen.add(pid)

    # Top by novelty
    for pid in df.head(n_top * 2)["paper_id"]:
        if pid not in seen and len(selected) < len(seen) + n_top:
            selected.append(pid)
            seen.add(pid)

    # Bottom by novelty
    for pid in df.tail(n_bottom * 2)["paper_id"][::-1]:
        if pid not in seen and len([s for s in selected if s in seen]) < n_top + n_bottom + len(golden_titles or []):
            selected.append(pid)
            seen.add(pid)

    # Middle (random)
    middle = df.iloc[len(df) // 4: 3 * len(df) // 4].sample(min(n_middle * 2, len(df) // 2), random_state=42)
    for pid in middle["paper_id"]:
        if pid not in seen and len(selected) < len(seen) + n_top + n_bottom + n_middle:
            selected.append(pid)
            seen.add(pid)

    return selected


# ============================================================================
# FILE 2: ARTICLES (5234 papers with metadata + novelty)
# ============================================================================
def build_articles():
    print("Building embeddings_01_articles.xlsx ...")
    subset = pd.read_parquet(EXP1 / "subset.parquet")
    metrics = pd.read_parquet(EXP1 / "metrics_v3.parquet")

    # Merge
    df = subset.merge(metrics[["paper_id", "p_nn_novelty", "p_cos_dist", "D_M", "R", "C",
                                "BPI_original", "BPI_v5_DM_only"]],
                      left_on="acl_id", right_on="paper_id", how="left")

    # Compute percentiles
    for col in ["p_nn_novelty", "D_M", "BPI_original"]:
        df[f"{col}_pct"] = df[col].rank(pct=True) * 100

    # Build URL: ACL ID -> aclanthology URL
    def make_url(row):
        acl_id = row.get("acl_id", "")
        ax_id = row.get("arxiv_id", "")
        if acl_id and isinstance(acl_id, str):
            return f"https://aclanthology.org/{acl_id}/"
        if ax_id and isinstance(ax_id, str):
            return f"https://arxiv.org/abs/{ax_id}"
        return ""
    df["URL"] = df.apply(make_url, axis=1)

    # Sort by novelty descending
    df = df.sort_values("p_nn_novelty", ascending=False, na_position="last")

    out_df = pd.DataFrame({
        "Title": df["title"],
        "Year": df["year"].astype("Int64"),
        "Venue": df["venue"],
        "Citations": df["citation_count"].astype("Int64"),
        "Influential cites": df["influential_citation_count"].astype("Int64"),
        "URL": df["URL"],
        "ACL ID": df["acl_id"],
        "arXiv ID": df["arxiv_id"],
        "Has full text": df["has_fulltext"],
        "kNN novelty (raw)": df["p_nn_novelty"].round(4),
        "kNN novelty percentile": df["p_nn_novelty_pct"].round(1),
        "Cosine dist to corpus center": df["p_cos_dist"].round(4),
        "Mahalanobis D_M": df["D_M"].round(4),
        "Mahalanobis percentile": df["D_M_pct"].round(1),
        "Rarity R": df["R"].round(4),
        "Coherence C": df["C"].round(4),
        "BPI original": df["BPI_original"].round(4),
        "BPI percentile": df["BPI_original_pct"].round(1),
    })

    explanation = pd.DataFrame({
        "Plain language explanation": [
            "This file lists all 5,234 papers from the EMBEDDINGS topic of the ACL Anthology",
            "(papers about word/sentence/document embeddings, representation learning).",
            "",
            "Papers are sorted by 'kNN novelty' (most novel first). High novelty = paper",
            "introduces ideas different from prior papers in the same field.",
            "",
            "COLUMNS EXPLAINED:",
            "• Title, Year, Venue, Citations: standard metadata",
            "• URL: clickable link to the paper on aclanthology.org or arxiv.org",
            "• kNN novelty (raw): mean cosine distance to the 10 nearest prior papers in idea space.",
            "  Higher = more novel. Range typically 0.05 to 0.4.",
            "• kNN novelty percentile: percentile rank among all 5,234 papers (100% = most novel).",
            "• Cosine dist to corpus center: how far the paper is from the average paper.",
            "• Mahalanobis D_M: distance from corpus mean accounting for idea correlations.",
            "• Rarity R: -Sum(idea_weight * log(idea_prevalence)) — high if paper uses rare ideas.",
            "• Coherence C: Sum(idea_weight^2) — high if paper concentrates on a few ideas.",
            "• BPI original: R * C * (1/density) — the breakthrough potential index from the paper.",
            "",
            "USEFUL FILTERS:",
            "• Sort by 'kNN novelty percentile' descending = most novel papers",
            "• Filter 'Citations > 1000' AND 'kNN novelty percentile > 90' = validated breakthroughs",
            "• Filter 'kNN novelty percentile < 20' = candidate 'unoriginal' papers",
            "",
            f"Total papers: {len(out_df):,}",
            f"Papers with citations: {df['citation_count'].notna().sum():,}",
            f"Year range: {int(df['year'].min())} - {int(df['year'].max())}",
        ]
    })

    out_path = OUT / "embeddings_01_articles.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        explanation.to_excel(w, sheet_name="What is this", index=False)
        out_df.to_excel(w, sheet_name="All articles", index=False)
        for sheet in w.sheets.values():
            for col in sheet.columns:
                max_len = max((len(str(c.value)) for c in col[:30] if c.value), default=10)
                sheet.column_dimensions[col[0].column_letter].width = min(max_len + 2, 70)
    print(f"  saved: {out_path}")
    return df  # for use in next files


# ============================================================================
# FILE 3: CHUNKS -> IDEAS (live extraction)
# ============================================================================
def build_chunks_to_ideas(articles_df, n_papers=150):
    print("Building embeddings_02_chunks_to_ideas.xlsx ...")

    # Load golden set
    with open(EXP1 / "golden_results.json") as f:
        gold = json.load(f)
    golden_titles = [p["title_query"] for tier in ["high", "medium", "low"] for p in gold.get(tier, [])]

    # Load metrics for picking interesting papers
    metrics = pd.read_parquet(EXP1 / "metrics_v3.parquet")
    selected_pids = select_sample_papers(metrics, n_top=50, n_bottom=40, n_middle=60,
                                          golden_titles=golden_titles)
    selected_pids = selected_pids[:n_papers]

    sample_df = articles_df[articles_df["acl_id"].isin(selected_pids)].copy()
    print(f"    Selected {len(sample_df)} papers for re-extraction")

    # Re-extract
    results = asyncio.run(extract_examples(sample_df, max_chunks_per_paper=2))
    print(f"    Got {len(results)} chunk extractions")

    # Build wide table
    rows = []
    for r in results:
        title = r["title"][:120]
        ideas_str = "\n".join([f"• {i}" for i in r["ideas"]]) if r["ideas"] else "(no ideas extracted)"
        # Truncate chunk for readability
        chunk_preview = r["chunk_text"][:1800]
        if len(r["chunk_text"]) > 1800:
            chunk_preview += f"\n... [truncated, total {len(r['chunk_text']):,} chars]"
        # Get citation
        m = sample_df[sample_df["acl_id"] == r["acl_id"]]
        cit = int(m.iloc[0]["citation_count"]) if len(m) and pd.notna(m.iloc[0]["citation_count"]) else None
        year = int(m.iloc[0]["year"]) if len(m) else None
        url = m.iloc[0]["URL"] if len(m) else ""
        # Position label
        ci = r["chunk_idx"]
        total = r["total_chunks"]
        if ci == 0:
            pos = "Intro/start"
        elif ci == total - 1:
            pos = "Conclusion/end"
        elif ci == total // 2:
            pos = "Methods/middle"
        else:
            pos = f"Section {ci+1}"
        rows.append({
            "Paper title": title,
            "Year": year,
            "Citations": cit,
            "URL": url,
            "Chunk position": pos,
            "Chunk index": f"{r['chunk_idx']+1} of {r['total_chunks']}",
            "Chunk text (truncated to 1800 chars)": chunk_preview,
            "Number of ideas extracted": len(r["ideas"]),
            "Extracted ideas (one per line)": ideas_str,
        })

    df_out = pd.DataFrame(rows)
    df_out = df_out.sort_values(["Citations", "Paper title"], ascending=[False, True])

    explanation = pd.DataFrame({
        "How idea extraction works": [
            "STEP 2 of the pipeline: chunk-by-chunk LLM extraction of conceptual ideas.",
            "",
            "INPUT: paper full text (typical 30K-100K chars)",
            "PROCESS: split into ~12K-char chunks by paragraph boundaries (~3-8 chunks per paper).",
            "         For EACH chunk, send to gpt-5.4-nano with this prompt:",
            "",
            '         "Extract novel ideas from this scientific paper excerpt.',
            '          - Return JSON: {ideas: [idea1, idea2, ...]}',
            '          - 0-5 ideas, each 5-20 words, specific',
            '          - Focus on novel methods, techniques, models, findings',
            '          - Skip background, related work, references"',
            "",
            "OUTPUT: a list of ~3-5 short conceptual ideas per chunk.",
            "",
            "WHY THIS WORKS:",
            "• The LLM acts as a conceptual abstraction layer — turns 'we apply skip-gram with negative sampling'",
            "  into a clean idea like 'skip-gram word vectors with negative sampling'.",
            "• Different papers describing the same method get similar idea texts -> same embedding cluster.",
            "• Background/method/results/conclusion get processed separately because they're in different chunks.",
            "",
            "COST: ~$0.001 per paper at gpt-5.4-nano pricing ($0.10/M in, $0.40/M out).",
            "",
            "WHAT YOU SEE BELOW:",
            f"  {len(df_out)} chunk extractions from {df_out['Paper title'].nunique()} papers, including:",
            "  • All golden-set breakthrough papers (GloVe, Word2Vec, CNN-Sentence, etc.)",
            "  • Top 50 by novelty score",
            "  • Bottom 40 by novelty score",
            "  • 60 random middle-tier papers",
            "",
            "  Look at the ideas to verify the LLM is extracting meaningful conceptual content,",
            "  not just keywords or summaries.",
        ]
    })

    out_path = OUT / "embeddings_02_chunks_to_ideas.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        explanation.to_excel(w, sheet_name="How extraction works", index=False)
        df_out.to_excel(w, sheet_name="Chunk to ideas examples", index=False)
        # Auto-width with text wrapping for chunk text
        from openpyxl.styles import Alignment
        for sheet_name, sheet in w.sheets.items():
            for col in sheet.columns:
                col_letter = col[0].column_letter
                header_text = str(col[0].value or "")
                if "Chunk text" in header_text:
                    sheet.column_dimensions[col_letter].width = 90
                    for cell in col[1:]:
                        cell.alignment = Alignment(wrap_text=True, vertical="top")
                elif "Extracted ideas" in header_text or "Paper title" in header_text or "URL" in header_text:
                    sheet.column_dimensions[col_letter].width = 60
                    for cell in col[1:]:
                        cell.alignment = Alignment(wrap_text=True, vertical="top")
                else:
                    max_len = max((len(str(c.value)) for c in col[:30] if c.value), default=10)
                    sheet.column_dimensions[col_letter].width = min(max_len + 2, 40)
            # Set row heights for chunk-to-ideas sheet
            if sheet_name == "Chunk to ideas examples":
                for row_idx in range(2, len(df_out) + 2):
                    sheet.row_dimensions[row_idx].height = 200
    print(f"  saved: {out_path}")


# ============================================================================
# FILE 4: BASE IDEAS (clusters)
# ============================================================================
def build_base_ideas():
    print("Building embeddings_03_base_ideas.xlsx ...")
    with open(EXP1 / "cluster_descriptions.json") as f:
        cluster_desc = json.load(f)
    with open(EXP1 / "cluster_labels.json") as f:
        cluster_labels = json.load(f)
    with open(EXP1 / "cluster_config.json") as f:
        cluster_config = json.load(f)
    with open(EXP1 / "spectra_paper_ids.json") as f:
        spectra_pids = json.load(f)
    spectra = np.load(EXP1 / "spectra.npy")

    subset = pd.read_parquet(EXP1 / "subset.parquet")
    pid_to_title = dict(zip(subset["acl_id"], subset["title"]))
    pid_to_cit = dict(zip(subset["acl_id"], subset["citation_count"]))
    pid_to_year = dict(zip(subset["acl_id"], subset["year"]))

    # For each cluster (base idea), find:
    # - cluster size (how many ideas in it)
    # - representative description
    # - top papers using this idea (by spectrum weight)
    # - mean weight across papers
    # - distinct papers count
    labels = np.array(cluster_labels)
    n_clusters = spectra.shape[1]

    rows = []
    for cid in range(n_clusters):
        # Top papers by weight on this cluster
        weights = spectra[:, cid]
        top_idx = np.argsort(weights)[-10:][::-1]
        top_papers = []
        for i in top_idx:
            pid = spectra_pids[i]
            title = pid_to_title.get(pid, "")[:60]
            cit = pid_to_cit.get(pid)
            year = pid_to_year.get(pid)
            cit_str = f"{int(cit)}cit" if pd.notna(cit) else "?cit"
            year_str = f"{int(year)}" if pd.notna(year) else "?"
            w = weights[i]
            top_papers.append(f"[{w:.2f}] {title} ({year_str}, {cit_str})")
        rows.append({
            "Cluster ID": cid,
            "Description (representative idea)": cluster_desc.get(str(cid), ""),
            "Cluster size (ideas in cluster)": int((labels == cid).sum()),
            "Mean weight across papers": round(float(weights.mean()), 4),
            "Max weight": round(float(weights.max()), 4),
            "% papers with weight>0.3": round(float((weights > 0.3).mean()) * 100, 1),
            "Top 10 papers (weight, title, year, citations)": "\n".join(top_papers),
        })

    df_clusters = pd.DataFrame(rows).sort_values("Mean weight across papers", ascending=False)

    explanation = pd.DataFrame({
        "What are base ideas": [
            f"STEP 3-4 of the pipeline: clustered {62000:,} extracted ideas into {n_clusters} 'base ideas'.",
            "",
            f"PROCESS: K-Means clustering (K={cluster_config['n_clusters']}) on the 62K idea embeddings",
            f"         (text-embedding-3-small, 1536 dims, reduced to 256 via PCA).",
            f"         Silhouette score: {cluster_config['silhouette']:.3f}",
            "",
            "WHY THIS MATTERS:",
            "Each paper's full set of ideas gets reduced to a fixed-length 'idea spectrum' — a vector of",
            "weights showing how much each base idea is present in the paper.",
            "This makes papers comparable: instead of 'paper A has 12 ideas and paper B has 8',",
            "we get 'paper A has weight 0.73 on cluster #5 and paper B has 0.41 on cluster #5'.",
            "",
            "WHAT YOU SEE BELOW:",
            f"All {n_clusters} base idea clusters (sorted by mean weight = popularity).",
            "• Description: the idea text closest to the cluster centroid (most representative).",
            "• Cluster size: how many of the 62K ideas got grouped here.",
            "• Mean weight: average importance of this idea across all papers (popularity).",
            "• Top 10 papers: papers that most strongly express this idea (with weights, year, cites).",
            "",
            "INTERPRETATION:",
            "• High 'mean weight' = mainstream idea appearing in many papers (e.g. 'attention mechanism').",
            "• Low 'mean weight' = niche/rare idea (e.g. specific to a few papers).",
            "• A NOVEL paper has high weights on RARE clusters AND focused (high coherence).",
        ]
    })

    out_path = OUT / "embeddings_03_base_ideas.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        explanation.to_excel(w, sheet_name="What are base ideas", index=False)
        df_clusters.to_excel(w, sheet_name="100 base ideas", index=False)
        from openpyxl.styles import Alignment
        for sheet_name, sheet in w.sheets.items():
            for col in sheet.columns:
                col_letter = col[0].column_letter
                header = str(col[0].value or "")
                if "Top 10" in header or "Description" in header:
                    sheet.column_dimensions[col_letter].width = 80
                    for cell in col[1:]:
                        cell.alignment = Alignment(wrap_text=True, vertical="top")
                else:
                    max_len = max((len(str(c.value)) for c in col[:30] if c.value), default=10)
                    sheet.column_dimensions[col_letter].width = min(max_len + 2, 50)
            if sheet_name == "100 base ideas":
                for row_idx in range(2, n_clusters + 2):
                    sheet.row_dimensions[row_idx].height = 180
    print(f"  saved: {out_path}")


# ============================================================================
# FILE 5: PAPER -> TOP IDEAS (with weights)
# ============================================================================
def build_paper_idea_weights(articles_df, n_papers=200):
    print("Building embeddings_04_paper_idea_weights.xlsx ...")
    with open(EXP1 / "cluster_descriptions.json") as f:
        cluster_desc = json.load(f)
    with open(EXP1 / "spectra_paper_ids.json") as f:
        spectra_pids = json.load(f)
    spectra = np.load(EXP1 / "spectra.npy")

    pid_to_idx = {pid: i for i, pid in enumerate(spectra_pids)}

    subset = pd.read_parquet(EXP1 / "subset.parquet")
    metrics = pd.read_parquet(EXP1 / "metrics_v3.parquet")

    # Load golden set
    with open(EXP1 / "golden_results.json") as f:
        gold = json.load(f)
    golden_titles = [p["title_query"] for tier in ["high", "medium", "low"] for p in gold.get(tier, [])]

    # Pick representative papers: golden + top novelty + bottom + random
    selected = select_sample_papers(metrics, n_top=70, n_bottom=50, n_middle=70,
                                     golden_titles=golden_titles)
    selected = [p for p in selected if p in pid_to_idx][:n_papers]
    print(f"    Selected {len(selected)} papers")

    rows = []
    for pid in selected:
        idx = pid_to_idx[pid]
        weights = spectra[idx]
        # Top 10 ideas
        top10 = np.argsort(weights)[-10:][::-1]
        ideas_str = "\n".join([
            f"[{weights[c]:.3f}] (cluster #{c}) {cluster_desc.get(str(c), '')[:75]}"
            for c in top10
        ])
        # Lookup metadata
        m = subset[subset["acl_id"] == pid]
        if len(m) == 0:
            continue
        row = m.iloc[0]
        cit = int(row["citation_count"]) if pd.notna(row["citation_count"]) else None
        year = int(row["year"]) if pd.notna(row["year"]) else None
        # Get metric values
        mr = metrics[metrics["paper_id"] == pid]
        nov_pct = float(mr.iloc[0]["p_nn_novelty"]) if len(mr) and pd.notna(mr.iloc[0].get("p_nn_novelty")) else None
        # Compute idea concentration
        coherence = float(np.sum(weights ** 2))
        n_active = int((weights > 0.3).sum())
        rows.append({
            "Title": row["title"][:120],
            "Year": year,
            "Citations": cit,
            "URL": (f"https://aclanthology.org/{pid}/" if pid else ""),
            "kNN novelty score": round(nov_pct, 4) if nov_pct is not None else None,
            "# ideas active (weight>0.3)": n_active,
            "Coherence (sum of squared weights)": round(coherence, 3),
            "Top 10 base ideas (weight, cluster ID, description)": ideas_str,
        })

    df_out = pd.DataFrame(rows).sort_values("Citations", ascending=False, na_position="last")

    explanation = pd.DataFrame({
        "What is a paper's 'idea spectrum'": [
            "STEP 5 of the pipeline: each paper gets reduced to a vector of weights over the 100 base ideas.",
            "",
            "FORMULA: s_i(paper) = mean over paper's ideas of cosine_similarity(idea_embedding, base_idea_centroid_i)",
            "",
            "RESULT: each paper has a 100-dim vector. Each component i = how much base idea #i is 'present'",
            "        in this paper (range typically 0 to 1). Most components are small (~0.1-0.3),",
            "        a few are high (>0.5) for the paper's main themes.",
            "",
            "WHAT YOU SEE BELOW:",
            f"For {len(df_out)} representative papers, the TOP 10 base ideas with their weights.",
            "Papers selected from: golden set + most novel + least novel + random middle-tier.",
            "Sorted by citations (most cited first).",
            "",
            "INTERPRETATION TIPS:",
            "• Look at well-known papers (GloVe, BERT, etc.) — their top ideas should make sense.",
            "• High coherence + a few rare ideas = candidate breakthrough.",
            "• Low coherence (many small weights) = paper covers many topics shallowly.",
            "• Compare papers in the same year — which ones express RARE ideas at high weight?",
            "",
            "FORMAT: '[0.473] (cluster #14) word2vec skip-gram with negative sampling'",
            "        means this paper has weight 0.473 on cluster #14, whose top idea text is shown.",
        ]
    })

    out_path = OUT / "embeddings_04_paper_idea_weights.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        explanation.to_excel(w, sheet_name="What is idea spectrum", index=False)
        df_out.to_excel(w, sheet_name="Papers and their top ideas", index=False)
        from openpyxl.styles import Alignment
        for sheet_name, sheet in w.sheets.items():
            for col in sheet.columns:
                col_letter = col[0].column_letter
                header = str(col[0].value or "")
                if "Top 10" in header:
                    sheet.column_dimensions[col_letter].width = 100
                    for cell in col[1:]:
                        cell.alignment = Alignment(wrap_text=True, vertical="top")
                elif "Title" in header or "URL" in header:
                    sheet.column_dimensions[col_letter].width = 50
                else:
                    max_len = max((len(str(c.value)) for c in col[:30] if c.value), default=10)
                    sheet.column_dimensions[col_letter].width = min(max_len + 2, 30)
            if sheet_name == "Papers and their top ideas":
                for row_idx in range(2, len(df_out) + 2):
                    sheet.row_dimensions[row_idx].height = 220
    print(f"  saved: {out_path}")


# ============================================================================
# FILE 6: METRICS + GOLDEN SET VALIDATION
# ============================================================================
def build_metrics_validation():
    print("Building embeddings_05_metrics_explained.xlsx ...")
    metrics = pd.read_parquet(EXP1 / "metrics_v3.parquet")
    subset = pd.read_parquet(EXP1 / "subset.parquet")
    df = metrics.merge(subset[["acl_id", "venue", "arxiv_id"]],
                        left_on="paper_id", right_on="acl_id", how="left")
    with open(EXP1 / "golden_results.json") as f:
        gold = json.load(f)

    # Compute correlations
    from scipy.stats import spearmanr
    df_test = df[df["year"] >= 2019].dropna(subset=["citation_count"]).copy()
    log_cit = np.log1p(df_test["citation_count"])

    metric_cols = ["p_nn_novelty", "p_cos_dist", "D_M", "R", "C", "rho",
                   "BPI_original", "BPI_v5_DM_only", "BPI_v6_DM_C", "BPI_v8_logDM_rho"]

    correl_rows = []
    for col in metric_cols:
        if col in df_test.columns:
            v = df_test[col].dropna()
            cit_v = np.log1p(df_test.loc[v.index, "citation_count"])
            r, p = spearmanr(v, cit_v)
            correl_rows.append({
                "Metric": col,
                "Description": {
                    "p_nn_novelty": "Mean cosine distance to 10 nearest prior papers (paper embedding space)",
                    "p_cos_dist": "Cosine distance from paper to corpus center",
                    "D_M": "Mahalanobis distance from corpus mean (idea spectrum)",
                    "R": "Rarity: -Sum(weight * log(prevalence))",
                    "C": "Coherence: Sum(weight^2)",
                    "rho": "Local density (1 / mean kNN distance)",
                    "BPI_original": "R * C * (1/density)  ← original formula from paper",
                    "BPI_v5_DM_only": "Just D_M (Mahalanobis)",
                    "BPI_v6_DM_C": "D_M * C",
                    "BPI_v8_logDM_rho": "log(D_M) / density",
                }.get(col, ""),
                "Spearman r vs log(citations)": round(float(r), 4),
                "p-value": f"{p:.2e}",
                "n": int(len(v)),
            })
    correl_df = pd.DataFrame(correl_rows).sort_values("Spearman r vs log(citations)", ascending=False)

    # Golden set: high tier
    def make_golden_table(tier, expected):
        rows = []
        for p in gold.get(tier, []):
            pid = p.get("paper_id")
            mr = df[df["paper_id"] == pid]
            if len(mr) == 0: continue
            r = mr.iloc[0]
            for col in ["p_nn_novelty", "D_M", "BPI_original"]:
                v = r.get(col)
                if pd.notna(v):
                    pct = float(df[col].rank(pct=True).iloc[df.index.get_loc(mr.index[0])] * 100)
                    rows.append({
                        "Paper": p["title"][:80],
                        "Tier": tier.upper(),
                        "Year": int(r["year"]) if pd.notna(r["year"]) else None,
                        "Citations": int(p.get("citation_count", 0) or 0),
                        "Metric": col,
                        "Score (raw)": round(float(v), 4),
                        "Percentile rank": round(pct, 1),
                        "Expected": expected,
                    })
        return rows

    golden_rows = []
    golden_rows += make_golden_table("high", "HIGH (top 25%)")
    golden_rows += make_golden_table("medium", "MEDIUM")
    golden_rows += make_golden_table("low", "LOW (bottom 50%)")
    golden_df = pd.DataFrame(golden_rows)

    # Top 50 and Bottom 50 by p_nn_novelty
    df_sorted = df.dropna(subset=["p_nn_novelty"]).sort_values("p_nn_novelty", ascending=False)
    top50 = df_sorted.head(50)[["title", "year", "citation_count", "p_nn_novelty", "D_M", "BPI_original"]].copy()
    bot50 = df_sorted.tail(50)[::-1][["title", "year", "citation_count", "p_nn_novelty", "D_M", "BPI_original"]].copy()
    for d in (top50, bot50):
        d.columns = ["Title", "Year", "Citations", "kNN novelty", "Mahalanobis D_M", "BPI original"]
        for c in ["kNN novelty", "Mahalanobis D_M", "BPI original"]:
            d[c] = d[c].round(4)
        d["Year"] = d["Year"].astype("Int64")
        d["Citations"] = d["Citations"].astype("Int64")

    explanation = pd.DataFrame({
        "Metrics and validation": [
            "STEP 6-7 of the pipeline: compute metrics and validate against ground truth (citations, golden set).",
            "",
            "WHAT EACH METRIC TRIES TO MEASURE:",
            "• kNN novelty: 'distance from prior papers' — measured in paper-embedding space",
            "• Mahalanobis D_M: 'distance from corpus mean' (accounts for covariance structure)",
            "• Rarity R: 'how rare are this paper's ideas in the corpus?'",
            "• Coherence C: 'how focused is this paper on a few ideas?'",
            "• Density rho: '1/distance to nearest neighbors' — high = many similar papers nearby",
            "• BPI = Rarity * Coherence * 1/density — original formula",
            "",
            "VALIDATION TARGETS:",
            "1) Citation correlation (Spearman r). Higher r = better proxy for impact.",
            "2) Golden set: known breakthrough papers should rank in TOP 25% for HIGH tier.",
            "                application papers (LOW tier) should rank in BOTTOM 50%.",
            "",
            "RESULTS ON 5,234 EMBEDDINGS PAPERS (test set: papers from 2019+):",
            f"  • kNN novelty: r = +0.333 ← BEST single metric",
            f"  • BPI original: r = -0.179 ← FAILS (correctly identifies novelty, but novelty != citations)",
            f"  • D_M (Mahalanobis): r ≈ +0.06",
            "",
            "KEY INSIGHT: BPI ranks GloVe (34K cit) at the 14th percentile (LOW). This is CORRECT —",
            "GloVe is a great paper but it's an INCREMENTAL improvement on word2vec, not structurally novel.",
            "BPI measures novelty of IDEAS, not citation count.",
            "",
            "WHAT YOU SEE BELOW:",
            "  Sheet 'Metric correlations': all metrics ranked by their citation correlation.",
            "  Sheet 'Golden set': known papers and how they rank under each metric.",
            "  Sheet 'Top 50 by novelty': the 50 most novel papers (by kNN novelty).",
            "  Sheet 'Bottom 50 by novelty': the 50 LEAST novel papers (likely 'me-too' / application papers).",
        ]
    })

    out_path = OUT / "embeddings_05_metrics_explained.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        explanation.to_excel(w, sheet_name="Explanation", index=False)
        correl_df.to_excel(w, sheet_name="Metric correlations", index=False)
        golden_df.to_excel(w, sheet_name="Golden set", index=False)
        top50.to_excel(w, sheet_name="Top 50 by novelty", index=False)
        bot50.to_excel(w, sheet_name="Bottom 50 by novelty", index=False)
        for sheet in w.sheets.values():
            for col in sheet.columns:
                max_len = max((len(str(c.value)) for c in col[:30] if c.value), default=10)
                sheet.column_dimensions[col[0].column_letter].width = min(max_len + 2, 90)
    print(f"  saved: {out_path}")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    t0 = time.time()
    build_cross_topic_summary()
    articles_df = build_articles()
    build_chunks_to_ideas(articles_df, n_papers=150)
    build_base_ideas()
    build_paper_idea_weights(articles_df, n_papers=200)
    build_metrics_validation()
    print(f"\nAll XLSX files saved to: {OUT}")
    print(f"Total time: {time.time()-t0:.0f}s")
    for f in sorted(OUT.glob("*.xlsx")):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name}: {size_mb:.1f} MB")
