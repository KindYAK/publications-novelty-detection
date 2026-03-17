"""LLM-based idea extraction from text segments.

Implements f_LLM(c_j, w, p) — extracting a concise conceptual formulation
from a text segment given the full document context.
"""

from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    """Protocol for LLM API clients."""

    def complete(self, prompt: str) -> str: ...


IDEA_EXTRACTION_PROMPT = """Сформулируйте основную концептуальную идею данного текстового фрагмента \
с учётом контекста всего документа в виде одного краткого и максимально обобщённого утверждения. \
Сосредоточьтесь на концептуальной сути идеи, опуская частные детали, примеры и технические реализации.

Контекст документа:
{document_context}

Текстовый фрагмент:
{segment_text}

Основная идея:"""

SUMMARIZE_IDEAS_PROMPT = """Обобщите приведённые идейные формулировки в одно краткое и максимально \
абстрактное утверждение, отражающее их общую концептуальную суть. Исключите частные детали \
и сохраните только наиболее общий смысл, объединяющий все формулировки.

Формулировки:
{ideas}

Обобщённая идея:"""


def extract_idea(segment: str, document_context: str, llm: LLMClient) -> str:
    """Extract the core conceptual idea from a text segment.

    Args:
        segment: The text segment to analyze.
        document_context: Full document text for context.
        llm: LLM client implementing the complete() method.

    Returns:
        Concise textual formulation of the segment's core idea.
    """
    prompt = IDEA_EXTRACTION_PROMPT.format(
        document_context=document_context[:4000],  # truncate long docs
        segment_text=segment,
    )
    return llm.complete(prompt).strip()


def summarize_idea_cluster(ideas: list[str], llm: LLMClient) -> str:
    """Summarize a cluster of similar idea formulations into one abstract statement.

    Args:
        ideas: List of idea formulations in the cluster.
        llm: LLM client.

    Returns:
        Single summarized idea formulation.
    """
    ideas_text = "\n".join(f"- {idea}" for idea in ideas)
    prompt = SUMMARIZE_IDEAS_PROMPT.format(ideas=ideas_text)
    return llm.complete(prompt).strip()
