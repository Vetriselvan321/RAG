"""RAGAS-style RAG evaluation using LLM-as-judge — Phase 5.

Four metrics mirror the RAGAS paper, implemented with the project's existing
Groq/Gemini provider so no additional API keys or packages are required.

  Faithfulness      — are answer claims grounded in the retrieved context?
  Context Relevance — are retrieved chunks relevant to the query?
  Context Recall    — does the context cover the ground-truth answer?
  Answer Relevance  — does the answer actually address the question?

Each metric returns a float in [0, 1]. Higher is better.
The LLM is asked to output structured JSON; the result is parsed and scored.
If the LLM returns malformed output the metric falls back to 0.0.
"""

import json
import re

import numpy as np

from .models import EvalSample, EvalResult, EvalReport
from .generator import Generator
from .embeddings import Embedder


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two un-normalised vectors."""
    va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


def _parse_json(text: str) -> dict:
    """Extract and parse the first JSON object from an LLM response.

    Handles markdown code fences (```json ... ```) and extra prose around
    the JSON object.
    """
    text = re.sub(r"```(?:json)?\s*", "", text).strip("`").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)


class RagasEvaluator:
    """LLM-as-judge evaluator implementing the four core RAGAS metrics.

    Args:
        generator: Existing Generator instance (reuses configured provider).
                   Defaults to creating a new one from environment.
        embedder:  Existing Embedder instance.
                   Defaults to creating a new one from environment.
    """

    def __init__(
        self,
        generator: Generator | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self._gen = generator or Generator()
        self._emb = embedder or Embedder()

    # ------------------------------------------------------------------
    # Internal LLM call — temperature=0 for deterministic scoring
    # ------------------------------------------------------------------

    def _llm(self, prompt: str) -> str:
        """Call the configured LLM at temperature=0 and return the raw text."""
        if self._gen.provider == "groq":
            resp = self._gen._client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return resp.choices[0].message.content

        else:  # gemini
            from google.genai import types as genai_types
            resp = self._gen._client.models.generate_content(
                model=self._gen.model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(temperature=0),
            )
            return resp.text

    # ------------------------------------------------------------------
    # Four RAGAS metrics
    # ------------------------------------------------------------------

    def faithfulness(self, answer: str, contexts: list[str]) -> float:
        """Fraction of answer statements that are grounded in the context.

        The LLM extracts individual factual claims from the answer and marks
        each as supported/unsupported by the context.
        Score = supported_count / total_count
        """
        if not answer.strip() or not contexts:
            return 0.0

        context_text = "\n\n".join(contexts)
        prompt = f"""You are a RAG faithfulness evaluator.

Context:
{context_text}

Answer:
{answer}

Extract each individual factual statement from the answer.
For each statement decide if it can be directly inferred from the context.

Respond with ONLY valid JSON — no explanation, no markdown:
{{"statements": [{{"statement": "...", "supported": true}}, {{"statement": "...", "supported": false}}]}}"""

        try:
            data = _parse_json(self._llm(prompt))
            stmts = data.get("statements", [])
            if not stmts:
                return 0.0
            return round(sum(1 for s in stmts if s.get("supported")) / len(stmts), 4)
        except Exception:
            return 0.0

    def context_relevance(self, query: str, contexts: list[str]) -> float:
        """Fraction of context sentences that are relevant to the query.

        The LLM counts relevant vs total sentences in the retrieved context.
        Score = relevant_sentence_count / total_sentence_count
        """
        if not contexts:
            return 0.0

        context_text = "\n\n".join(contexts)
        prompt = f"""You are a RAG context relevance evaluator.

Query: {query}

Context:
{context_text}

Count the total number of sentences in the context.
Then count how many of those sentences are useful for answering the query.

Respond with ONLY valid JSON — no explanation, no markdown:
{{"relevant_count": 3, "total_count": 5}}"""

        try:
            data = _parse_json(self._llm(prompt))
            relevant = int(data.get("relevant_count", 0))
            total = int(data.get("total_count", 1))
            if total == 0:
                return 0.0
            return round(min(1.0, relevant / total), 4)
        except Exception:
            return 0.0

    def context_recall(self, ground_truth: str, contexts: list[str]) -> float:
        """Fraction of ground-truth claims that can be found in the context.

        The LLM breaks the ground truth into claims and checks each against
        the retrieved context.
        Score = in_context_count / total_claim_count
        """
        if not ground_truth.strip() or not contexts:
            return 0.0

        context_text = "\n\n".join(contexts)
        prompt = f"""You are a RAG context recall evaluator.

Ground Truth Answer:
{ground_truth}

Retrieved Context:
{context_text}

Break the ground truth into individual factual claims.
For each claim decide if it can be found in or inferred from the context.

Respond with ONLY valid JSON — no explanation, no markdown:
{{"claims": [{{"claim": "...", "in_context": true}}, {{"claim": "...", "in_context": false}}]}}"""

        try:
            data = _parse_json(self._llm(prompt))
            claims = data.get("claims", [])
            if not claims:
                return 0.0
            return round(sum(1 for c in claims if c.get("in_context")) / len(claims), 4)
        except Exception:
            return 0.0

    def answer_relevance(self, query: str, answer: str) -> float:
        """Mean cosine similarity of reverse-generated questions to the original query.

        The LLM generates N questions that the answer would best respond to.
        We embed those questions and the original query; mean cosine similarity
        measures how well the answer addresses what was actually asked.
        Score ∈ [0, 1] (higher = answer is more on-topic)
        """
        if not answer.strip():
            return 0.0

        prompt = f"""You are a RAG answer relevance evaluator.

Answer: {answer}

Generate exactly 3 questions that this answer would best respond to.

Respond with ONLY valid JSON — no explanation, no markdown:
{{"questions": ["question 1", "question 2", "question 3"]}}"""

        try:
            data = _parse_json(self._llm(prompt))
            questions = data.get("questions", [])
            if not questions:
                return 0.0

            q_vec = self._emb.embed_query(query)
            gen_vecs = self._emb.embed_texts(questions)
            similarities = [_cosine(q_vec, gv) for gv in gen_vecs]
            return round(max(0.0, min(1.0, sum(similarities) / len(similarities))), 4)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Scoring API
    # ------------------------------------------------------------------

    def score_sample(self, sample: EvalSample) -> EvalResult:
        """Compute all four metrics for a single EvalSample."""
        return EvalResult(
            query=sample.query,
            faithfulness=self.faithfulness(sample.answer, sample.contexts),
            context_relevance=self.context_relevance(sample.query, sample.contexts),
            context_recall=self.context_recall(sample.ground_truth, sample.contexts),
            answer_relevance=self.answer_relevance(sample.query, sample.answer),
        )

    def evaluate(self, samples: list[EvalSample]) -> EvalReport:
        """Score every sample and aggregate into an EvalReport."""
        return EvalReport(results=[self.score_sample(s) for s in samples])
