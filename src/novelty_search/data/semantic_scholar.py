"""Semantic Scholar API client for paper search and citation data.

Uses the Academic Graph API (free, no key required for basic access).
Rate limit: ~1 req/sec unauthenticated, higher with API key.

API docs: https://api.semanticscholar.org/api-docs/
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.semanticscholar.org/graph/v1"

# Fields we always request
PAPER_FIELDS = ",".join([
    "title",
    "abstract",
    "year",
    "venue",
    "publicationDate",
    "citationCount",
    "influentialCitationCount",
    "externalIds",
    "s2FieldsOfStudy",
    "openAccessPdf",
])


@dataclass
class Paper:
    """A paper with metadata and citation info."""

    paper_id: str
    title: str
    abstract: str | None
    year: int | None
    venue: str
    publication_date: str | None
    citation_count: int
    influential_citation_count: int
    arxiv_id: str | None
    doi: str | None
    fields_of_study: list[str]
    open_access_url: str | None

    @classmethod
    def from_api(cls, data: dict) -> Paper:
        ext = data.get("externalIds") or {}
        fos = data.get("s2FieldsOfStudy") or []
        oa = data.get("openAccessPdf") or {}
        return cls(
            paper_id=data["paperId"],
            title=data.get("title") or "",
            abstract=data.get("abstract"),
            year=data.get("year"),
            venue=data.get("venue") or "",
            publication_date=data.get("publicationDate"),
            citation_count=data.get("citationCount") or 0,
            influential_citation_count=data.get("influentialCitationCount") or 0,
            arxiv_id=ext.get("ArXiv"),
            doi=ext.get("DOI"),
            fields_of_study=[f["category"] for f in fos],
            open_access_url=oa.get("url") or None,
        )

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "abstract": self.abstract,
            "year": self.year,
            "venue": self.venue,
            "publication_date": self.publication_date,
            "citation_count": self.citation_count,
            "influential_citation_count": self.influential_citation_count,
            "arxiv_id": self.arxiv_id,
            "doi": self.doi,
            "fields_of_study": self.fields_of_study,
            "open_access_url": self.open_access_url,
        }


@dataclass
class SemanticScholarClient:
    """Client for the Semantic Scholar Academic Graph API."""

    api_key: str | None = None
    request_interval: float = 1.1  # seconds between requests (respect rate limit)
    _last_request_time: float = field(default=0.0, repr=False)

    @property
    def _headers(self) -> dict:
        h = {"Accept": "application/json"}
        if self.api_key:
            h["x-api-key"] = self.api_key
        return h

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < self.request_interval:
            time.sleep(self.request_interval - elapsed)
        self._last_request_time = time.time()

    def _get(self, url: str, params: dict | None = None) -> dict:
        self._throttle()
        resp = requests.get(url, params=params, headers=self._headers, timeout=30)
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 5))
            logger.warning("Rate limited, waiting %d seconds", wait)
            time.sleep(wait)
            return self._get(url, params)
        resp.raise_for_status()
        return resp.json()

    def get_paper(self, paper_id: str) -> Paper:
        """Fetch a single paper by Semantic Scholar ID, DOI, or ArXiv ID.

        Args:
            paper_id: S2 paper ID, or "ArXiv:2106.09685", or "DOI:10.xxx/yyy"
        """
        data = self._get(f"{BASE_URL}/paper/{paper_id}", {"fields": PAPER_FIELDS})
        return Paper.from_api(data)

    def search_bulk(
        self,
        query: str,
        year_range: str | None = None,
        fields_of_study: list[str] | None = None,
        min_citation_count: int | None = None,
        max_results: int = 500,
    ) -> list[Paper]:
        """Bulk search for papers matching a query.

        Args:
            query: Search terms (matched against title + abstract).
            year_range: e.g. "2015-2020" or "2018-" or "-2020".
            fields_of_study: Filter by S2 fields, e.g. ["Computer Science"].
            min_citation_count: Only return papers with at least this many citations.
            max_results: Maximum papers to return (API pages at 1000).

        Returns:
            List of Paper objects.
        """
        params: dict = {
            "query": query,
            "fields": PAPER_FIELDS,
            "limit": min(1000, max_results),
        }
        if year_range:
            params["year"] = year_range
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)
        if min_citation_count is not None:
            params["minCitationCount"] = str(min_citation_count)

        papers: list[Paper] = []
        token: str | None = None

        while len(papers) < max_results:
            if token:
                params["token"] = token

            data = self._get(f"{BASE_URL}/paper/search/bulk", params)
            total = data.get("total", 0)
            batch = data.get("data") or []

            if not batch:
                break

            for item in batch:
                if len(papers) >= max_results:
                    break
                try:
                    papers.append(Paper.from_api(item))
                except (KeyError, TypeError) as e:
                    logger.warning("Skipping malformed paper: %s", e)

            token = data.get("token")
            if not token:
                break

            logger.info("Fetched %d / %d papers (total available: %d)", len(papers), max_results, total)

        return papers

    def search_by_arxiv_category(
        self,
        category_keywords: str,
        year_range: str = "2015-2020",
        min_citation_count: int = 0,
        max_results: int = 500,
    ) -> list[Paper]:
        """Search for papers in an arXiv-like category using keyword approximation.

        Semantic Scholar doesn't support arXiv categories directly, so we use
        keyword search + field-of-study filter as a proxy.

        Args:
            category_keywords: Keywords that approximate the category,
                e.g. "natural language processing" for cs.CL.
            year_range: Year filter.
            min_citation_count: Minimum citations.
            max_results: Max papers.
        """
        return self.search_bulk(
            query=category_keywords,
            year_range=year_range,
            fields_of_study=["Computer Science"],
            min_citation_count=min_citation_count,
            max_results=max_results,
        )
