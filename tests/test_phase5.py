"""
tests/test_phase5.py

Tests for Phase 5: RAGAS-style evaluation framework.

Run all Phase 5 tests:          pytest tests/test_phase5.py -v
Run only unit tests (no LLM):   pytest tests/test_phase5.py -v -k "not LLM and not Evaluator"
"""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.models import EvalSample, EvalResult, EvalReport


# ---------------------------------------------------------------------------
# EvalSample — pure unit tests
# ---------------------------------------------------------------------------

class TestEvalSample:

    def test_required_fields(self):
        s = EvalSample(query="q", ground_truth="gt")
        assert s.query == "q"
        assert s.ground_truth == "gt"

    def test_optional_fields_default(self):
        s = EvalSample(query="q", ground_truth="gt")
        assert s.answer == ""
        assert s.contexts == []

    def test_optional_fields_set(self):
        s = EvalSample(query="q", ground_truth="gt", answer="a", contexts=["c"])
        assert s.answer == "a"
        assert s.contexts == ["c"]

    def test_contexts_are_independent_per_instance(self):
        s1 = EvalSample(query="q1", ground_truth="g1")
        s2 = EvalSample(query="q2", ground_truth="g2")
        s1.contexts.append("x")
        assert s2.contexts == []


# ---------------------------------------------------------------------------
# EvalResult — pure unit tests
# ---------------------------------------------------------------------------

class TestEvalResult:

    def _result(self, f=0.8, cr=0.7, cc=0.6, ar=0.9):
        return EvalResult(
            query="test query",
            faithfulness=f,
            context_relevance=cr,
            context_recall=cc,
            answer_relevance=ar,
        )

    def test_mean_score_calculation(self):
        r = self._result(0.8, 0.7, 0.6, 0.9)
        assert abs(r.mean_score - (0.8 + 0.7 + 0.6 + 0.9) / 4) < 1e-4

    def test_mean_score_perfect(self):
        r = self._result(1.0, 1.0, 1.0, 1.0)
        assert r.mean_score == 1.0

    def test_mean_score_zero(self):
        r = self._result(0.0, 0.0, 0.0, 0.0)
        assert r.mean_score == 0.0

    def test_all_attributes_stored(self):
        r = self._result()
        assert r.faithfulness == 0.8
        assert r.context_relevance == 0.7
        assert r.context_recall == 0.6
        assert r.answer_relevance == 0.9


# ---------------------------------------------------------------------------
# EvalReport — pure unit tests
# ---------------------------------------------------------------------------

class TestEvalReport:

    def _report(self):
        return EvalReport(results=[
            EvalResult("q1", faithfulness=1.0, context_relevance=0.8,
                       context_recall=0.6, answer_relevance=0.9),
            EvalResult("q2", faithfulness=0.5, context_relevance=0.4,
                       context_recall=0.8, answer_relevance=0.7),
        ])

    def test_faithfulness_average(self):
        report = self._report()
        assert abs(report.faithfulness - (1.0 + 0.5) / 2) < 1e-4

    def test_context_relevance_average(self):
        report = self._report()
        assert abs(report.context_relevance - (0.8 + 0.4) / 2) < 1e-4

    def test_context_recall_average(self):
        report = self._report()
        assert abs(report.context_recall - (0.6 + 0.8) / 2) < 1e-4

    def test_answer_relevance_average(self):
        report = self._report()
        assert abs(report.answer_relevance - (0.9 + 0.7) / 2) < 1e-4

    def test_overall_is_mean_of_four_metrics(self):
        report = self._report()
        expected = (report.faithfulness + report.context_relevance
                    + report.context_recall + report.answer_relevance) / 4
        assert abs(report.overall - expected) < 1e-4

    def test_single_sample_report(self):
        report = EvalReport(results=[
            EvalResult("q", 0.5, 0.5, 0.5, 0.5)
        ])
        assert report.faithfulness == 0.5
        assert report.overall == 0.5

    def test_results_list_stored(self):
        report = self._report()
        assert len(report.results) == 2


# ---------------------------------------------------------------------------
# _parse_json helper — pure unit test
# ---------------------------------------------------------------------------

class TestParseJson:

    def test_plain_json(self):
        from rag.evaluator import _parse_json
        result = _parse_json('{"statements": [{"statement": "x", "supported": true}]}')
        assert result["statements"][0]["supported"] is True

    def test_json_with_markdown_fence(self):
        from rag.evaluator import _parse_json
        text = '```json\n{"claims": [{"claim": "y", "in_context": false}]}\n```'
        result = _parse_json(text)
        assert result["claims"][0]["in_context"] is False

    def test_json_with_prose_before(self):
        from rag.evaluator import _parse_json
        text = 'Here is my evaluation:\n{"relevant_count": 3, "total_count": 5}'
        result = _parse_json(text)
        assert result["total_count"] == 5

    def test_invalid_json_raises(self):
        from rag.evaluator import _parse_json
        with pytest.raises(Exception):
            _parse_json("not json at all")


# ---------------------------------------------------------------------------
# _cosine helper — pure unit test
# ---------------------------------------------------------------------------

class TestCosine:

    def test_identical_vectors_return_one(self):
        from rag.evaluator import _cosine
        v = [1.0, 2.0, 3.0]
        assert abs(_cosine(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors_return_zero(self):
        from rag.evaluator import _cosine
        assert abs(_cosine([1.0, 0.0], [0.0, 1.0])) < 1e-6

    def test_opposite_vectors_return_negative_one(self):
        from rag.evaluator import _cosine
        assert abs(_cosine([1.0, 0.0], [-1.0, 0.0]) - (-1.0)) < 1e-6

    def test_zero_vector_returns_zero(self):
        from rag.evaluator import _cosine
        assert _cosine([0.0, 0.0], [1.0, 2.0]) == 0.0


# ---------------------------------------------------------------------------
# RagasEvaluator — requires LLM API
# ---------------------------------------------------------------------------

def _llm_available() -> bool:
    try:
        from rag.generator import Generator
        Generator()
        return True
    except Exception:
        return False


requires_llm = pytest.mark.skipif(
    not _llm_available(),
    reason="LLM provider not configured (set LLM_PROVIDER + API key in .env)",
)


def _call_or_skip(fn):
    """Run fn(); skip gracefully on quota / auth errors."""
    try:
        return fn()
    except Exception as e:
        msg = str(e).lower()
        if any(k in msg for k in ("quota", "rate", "limit", "exhausted", "auth", "invalid", "key")):
            pytest.skip(f"LLM quota/auth: {e}")
        raise


@requires_llm
class TestRagasEvaluatorMetrics:

    @pytest.fixture(scope="class")
    def evaluator(self):
        from rag.evaluator import RagasEvaluator
        return RagasEvaluator()

    # ----- faithfulness -----

    def test_faithful_answer_scores_higher_than_hallucination(self, evaluator):
        context = ["Python was created by Guido van Rossum in 1991."]
        faithful = "Python was created by Guido van Rossum in 1991. [1]"
        hallucinated = "Python was invented by Microsoft in 2005 and first run on Windows."

        score_faithful = _call_or_skip(lambda: evaluator.faithfulness(faithful, context))
        score_hallucinated = _call_or_skip(lambda: evaluator.faithfulness(hallucinated, context))
        assert score_faithful > score_hallucinated

    def test_faithfulness_empty_answer_returns_zero(self, evaluator):
        assert evaluator.faithfulness("", ["some context"]) == 0.0

    def test_faithfulness_empty_context_returns_zero(self, evaluator):
        assert evaluator.faithfulness("some answer", []) == 0.0

    def test_faithfulness_returns_float_in_range(self, evaluator):
        context = ["The sky is blue."]
        answer = "The sky is blue."
        score = _call_or_skip(lambda: evaluator.faithfulness(answer, context))
        assert 0.0 <= score <= 1.0

    # ----- context_relevance -----

    def test_relevant_context_scores_higher(self, evaluator):
        query = "What programming language did Guido van Rossum create?"
        relevant = ["Python was created by Guido van Rossum."]
        irrelevant = ["The Eiffel Tower is located in Paris, France."]

        score_rel = _call_or_skip(lambda: evaluator.context_relevance(query, relevant))
        score_irr = _call_or_skip(lambda: evaluator.context_relevance(query, irrelevant))
        assert score_rel > score_irr

    def test_context_relevance_empty_returns_zero(self, evaluator):
        assert evaluator.context_relevance("query", []) == 0.0

    def test_context_relevance_returns_float_in_range(self, evaluator):
        score = _call_or_skip(
            lambda: evaluator.context_relevance("What is Python?", ["Python is a language."])
        )
        assert 0.0 <= score <= 1.0

    # ----- context_recall -----

    def test_context_covering_ground_truth_scores_higher(self, evaluator):
        ground_truth = "Python was created by Guido van Rossum in 1991."
        covering = ["Python is a language created by Guido van Rossum in 1991."]
        missing = ["Java was created by James Gosling at Sun Microsystems."]

        score_cov = _call_or_skip(lambda: evaluator.context_recall(ground_truth, covering))
        score_mis = _call_or_skip(lambda: evaluator.context_recall(ground_truth, missing))
        assert score_cov > score_mis

    def test_context_recall_empty_ground_truth_returns_zero(self, evaluator):
        assert evaluator.context_recall("", ["context"]) == 0.0

    def test_context_recall_returns_float_in_range(self, evaluator):
        score = _call_or_skip(
            lambda: evaluator.context_recall("Python is a language.", ["Python is a language."])
        )
        assert 0.0 <= score <= 1.0

    # ----- answer_relevance -----

    def test_on_topic_answer_scores_higher(self, evaluator):
        query = "What is Python used for?"
        on_topic = "Python is used for web development, data science, and machine learning."
        off_topic = "The French Revolution began in 1789 due to social inequality."

        score_on = _call_or_skip(lambda: evaluator.answer_relevance(query, on_topic))
        score_off = _call_or_skip(lambda: evaluator.answer_relevance(query, off_topic))
        assert score_on > score_off

    def test_answer_relevance_empty_returns_zero(self, evaluator):
        assert evaluator.answer_relevance("query", "") == 0.0

    def test_answer_relevance_returns_float_in_range(self, evaluator):
        score = _call_or_skip(
            lambda: evaluator.answer_relevance("What is Python?", "Python is a programming language.")
        )
        assert 0.0 <= score <= 1.0


@requires_llm
class TestRagasEvaluatorE2E:

    @pytest.fixture(scope="class")
    def evaluator(self):
        from rag.evaluator import RagasEvaluator
        return RagasEvaluator()

    def test_score_sample_returns_eval_result(self, evaluator):
        sample = EvalSample(
            query="Who created Python?",
            ground_truth="Python was created by Guido van Rossum in 1991.",
            answer="Python was created by Guido van Rossum in 1991. [1]",
            contexts=["Python is a language created by Guido van Rossum in 1991."],
        )
        result = _call_or_skip(lambda: evaluator.score_sample(sample))
        assert isinstance(result, EvalResult)
        assert result.query == sample.query

    def test_score_sample_all_metrics_in_range(self, evaluator):
        sample = EvalSample(
            query="What is Python?",
            ground_truth="Python is a high-level programming language.",
            answer="Python is a high-level programming language used for many tasks. [1]",
            contexts=["Python is a high-level programming language created by Guido van Rossum."],
        )
        result = _call_or_skip(lambda: evaluator.score_sample(sample))
        for metric in (result.faithfulness, result.context_relevance,
                       result.context_recall, result.answer_relevance):
            assert 0.0 <= metric <= 1.0

    def test_evaluate_returns_eval_report(self, evaluator):
        samples = [
            EvalSample(
                query="Who created Python?",
                ground_truth="Guido van Rossum created Python in 1991.",
                answer="Python was created by Guido van Rossum. [1]",
                contexts=["Python was created by Guido van Rossum in 1991."],
            ),
            EvalSample(
                query="What is machine learning?",
                ground_truth="Machine learning is a subset of AI that learns from data.",
                answer="Machine learning uses algorithms to learn patterns from data. [1]",
                contexts=["Machine learning uses statistical algorithms to learn from data."],
            ),
        ]
        report = _call_or_skip(lambda: evaluator.evaluate(samples))
        assert isinstance(report, EvalReport)
        assert len(report.results) == 2
        assert 0.0 <= report.overall <= 1.0

    def test_good_sample_scores_higher_than_bad_sample(self, evaluator):
        good = EvalSample(
            query="Who created Python?",
            ground_truth="Python was created by Guido van Rossum in 1991.",
            answer="Python was created by Guido van Rossum in 1991. [1]",
            contexts=["Python was created by Guido van Rossum in 1991."],
        )
        bad = EvalSample(
            query="Who created Python?",
            ground_truth="Python was created by Guido van Rossum in 1991.",
            answer="Python was invented by Microsoft in 2005 for Windows development.",
            contexts=["The Eiffel Tower is a landmark in Paris built in 1889."],
        )
        good_result = _call_or_skip(lambda: evaluator.score_sample(good))
        bad_result = _call_or_skip(lambda: evaluator.score_sample(bad))
        assert good_result.mean_score > bad_result.mean_score
