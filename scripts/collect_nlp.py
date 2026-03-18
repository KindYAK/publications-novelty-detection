"""Collect NLP papers from Semantic Scholar."""

import logging
import sys
sys.path.insert(0, "src")

from novelty_search.data.collector import collect_papers

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
df = collect_papers(config_name="nlp_core", output_dir="data/processed")
