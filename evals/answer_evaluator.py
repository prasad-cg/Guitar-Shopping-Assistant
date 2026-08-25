"""
Answer Evaluation Framework
Evaluates system responses against user queries using LLM-as-a-judge.

Scores responses on:
  - Relevance: Does the answer address what the user actually asked?
  - Completeness: Does it cover all aspects of the question?
  - Groundedness: Is it based on catalog data (not hallucinated)?
  - Helpfulness: Would this actually help a customer make a decision?
  - Clarity: Is the response well-structured and easy to understand?

Each dimension is scored 1-5, with an overall weighted average.
"""
import json
from typing import Dict, Any, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from utils.llm_setup import get_llm


EVALUATION_PROMPT = """You are an expert evaluator assessing the quality of an AI guitar shopping assistant's responses.

You will be given:
- The customer's question
- The system's response
- (Optionally) The retrieved context from the knowledge base

Score the response on each dimension from 1 to 5:

**Relevance** (1-5): Does the response directly address the customer's question?
- 5: Perfectly on-topic, addresses every part of the question
- 3: Partially relevant, misses some aspects
- 1: Completely off-topic or answers a different question

**Completeness** (1-5): Does it cover all aspects the customer needs?
- 5: Comprehensive, covers all angles (specs, recommendations, context)
- 3: Covers the basics but misses important details
- 1: Barely addresses the question, very superficial

**Groundedness** (1-5): Is the response based on the provided catalog/context?
- 5: Every claim is traceable to the knowledge base
- 3: Mix of grounded facts and general knowledge
- 1: Mostly hallucinated or invented information

**Helpfulness** (1-5): Would this help a real customer make a purchase decision?
- 5: Actionable, specific, directly useful for decision-making
- 3: Somewhat useful but too generic to act on
- 1: Not helpful, confusing, or misleading

**Clarity** (1-5): Is it well-structured and easy to understand?
- 5: Clear, well-organized, appropriate length
- 3: Understandable but could be better organized
- 1: Confusing, poorly structured, too verbose or too terse

Respond ONLY with valid JSON in this exact format:
{{
    "relevance": {{"score": <1-5>, "reason": "<one sentence>"}},
    "completeness": {{"score": <1-5>, "reason": "<one sentence>"}},
    "groundedness": {{"score": <1-5>, "reason": "<one sentence>"}},
    "helpfulness": {{"score": <1-5>, "reason": "<one sentence>"}},
    "clarity": {{"score": <1-5>, "reason": "<one sentence>"}},
    "overall_feedback": "<one paragraph summary of strengths and weaknesses>"
}}"""


class AnswerEvaluator:
    """Evaluates the quality of system responses against user queries."""

    # Weights for computing overall score
    WEIGHTS = {
        "relevance": 0.25,
        "completeness": 0.20,
        "groundedness": 0.25,
        "helpfulness": 0.20,
        "clarity": 0.10,
    }

    def __init__(self):
        self.llm = get_llm()

    def evaluate_single(
        self,
        user_query: str,
        system_response: str,
        retrieved_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single query-response pair.

        Returns:
            Dict with dimension scores, reasons, overall score (0-100), and feedback.
        """
        context_section = ""
        if retrieved_context:
            context_section = f"\n\nRetrieved Context (from knowledge base):\n{retrieved_context}"

        prompt = ChatPromptTemplate.from_messages([
            ("system", EVALUATION_PROMPT),
            ("user", "Customer Question:\n{user_query}\n\nSystem Response:\n{system_response}{context_section}"),
        ])

        chain = prompt | self.llm
        raw = chain.invoke({
            "user_query": user_query,
            "system_response": system_response,
            "context_section": context_section,
        })

        try:
            result = json.loads(raw.content)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            content = raw.content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                result = json.loads(content[start:end])
            else:
                return {
                    "error": "Failed to parse evaluation response",
                    "raw_response": raw.content,
                }

        # Compute overall weighted score (1-5 scale → 0-100 scale)
        total = 0.0
        for dim, weight in self.WEIGHTS.items():
            score = result.get(dim, {}).get("score", 3)
            total += score * weight
        overall_score = round((total / 5) * 100, 1)

        return {
            "dimensions": result,
            "overall_score": overall_score,
            "overall_feedback": result.get("overall_feedback", ""),
            "query": user_query,
        }

    def evaluate_batch(
        self,
        test_cases: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of test cases.

        Args:
            test_cases: List of dicts with keys "query" and "response",
                        optionally "context".

        Returns:
            Summary with individual results and aggregate scores.
        """
        results = []
        for case in test_cases:
            result = self.evaluate_single(
                user_query=case["query"],
                system_response=case["response"],
                retrieved_context=case.get("context"),
            )
            results.append(result)

        # Aggregate
        valid_results = [r for r in results if "error" not in r]
        if not valid_results:
            return {"results": results, "aggregate": None}

        avg_score = sum(r["overall_score"] for r in valid_results) / len(valid_results)

        dim_avgs = {}
        for dim in self.WEIGHTS:
            scores = [r["dimensions"][dim]["score"] for r in valid_results if dim in r.get("dimensions", {})]
            dim_avgs[dim] = round(sum(scores) / len(scores), 2) if scores else 0

        return {
            "results": results,
            "aggregate": {
                "overall_avg_score": round(avg_score, 1),
                "dimension_averages": dim_avgs,
                "total_evaluated": len(valid_results),
                "total_failed": len(results) - len(valid_results),
            },
        }
