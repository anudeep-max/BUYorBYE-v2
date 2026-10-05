"""Model router for selecting the appropriate LLM based on query complexity."""

from typing import Dict, Any, Optional, Tuple
import re

from src.analytics.logger import logger
from src.utils.config import settings


class ModelRouter:
    """Route queries to the configured LLM.

    BUYorBYE currently uses Groq as its single LLM provider.
    Complexity classification is retained so model routing can be expanded
    later without changing the rest of the application.
    """

    def __init__(self):
        self.complexity_threshold = 2

        self.complex_indicators = [
            r"\b(compare|comparison|difference|versus|vs|better|best)\b",
            r"\b(explain|why|how|what is|describe|details)\b",
            r"\b(recommend|suggest|advice|opinion|preference)\b",
            r"\b(multiple|several|many|various|different)\b",
            r"\b(pros|cons|advantages|disadvantages|benefits)\b",
            r"\b(review|rating|quality|reliability|durability)\b",
            r"\b(technical|specification|specs|features|capabilities)\b",
            r"\b(which|should|would|could|might)\b",
        ]

        self.simple_indicators = [
            r"^find\s+me\s+",
            r"^show\s+me\s+",
            r"^i\s+need\s+",
            r"^i\s+want\s+",
            r"^search\s+for\s+",
            r"under\s+[₹$€£]?\s*\d+",
            r"between\s+[₹$€£]?\s*\d+\s+and\s+[₹$€£]?\s*\d+",
        ]

    def classify_complexity(
        self,
        query: str,
        intent: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Classify a query as simple or complex."""

        query_lower = query.lower()

        simple_score = sum(
            1
            for pattern in self.simple_indicators
            if re.search(pattern, query_lower, re.IGNORECASE)
        )

        complex_score = sum(
            1
            for pattern in self.complex_indicators
            if re.search(pattern, query_lower, re.IGNORECASE)
        )

        if intent:
            intent_type = intent.get("type", "")

            if intent_type in [
                "comparison",
                "recommendation",
                "explanation",
            ]:
                complex_score += 2
            elif intent_type == "simple_search":
                simple_score += 1

            tools_needed = intent.get("tools_needed", [])
            if len(tools_needed) > 1:
                complex_score += 1

        word_count = len(query.split())

        if word_count > 15:
            complex_score += 1
        elif word_count < 5:
            simple_score += 1

        if complex_score >= self.complexity_threshold:
            return "complex"

        if simple_score > 0 and complex_score == 0:
            return "simple"

        return "simple"

    def select_model(
        self,
        query: str,
        intent: Optional[Dict[str, Any]] = None,
        force_model: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Select the configured Groq model.

        Returns:
            Tuple of (model_name, complexity_level)
        """

        if force_model:
            return force_model, "forced"

        complexity = self.classify_complexity(query, intent)

        provider = settings.llm_provider.lower()

        if provider != "groq":
            logger.warning(
                "BUYorBYE is configured for Groq, but provider is '%s'. "
                "Using configured model '%s'.",
                provider,
                settings.llm_model,
            )

        selected_model = settings.llm_model

        logger.debug(
            "Model routing: query='%s...' -> complexity=%s -> provider=%s -> model=%s",
            query[:50],
            complexity,
            provider,
            selected_model,
        )

        return selected_model, complexity

    def get_cost_savings_estimate(
        self,
        simple_queries: int,
        complex_queries: int,
    ) -> Dict[str, Any]:
        """Return routing statistics.

        Groq model pricing can vary by model/provider configuration, so this
        method intentionally avoids inventing cost estimates.
        """

        total_queries = simple_queries + complex_queries

        return {
            "simple_queries": simple_queries,
            "complex_queries": complex_queries,
            "total_queries": total_queries,
            "provider": settings.llm_provider,
            "model": settings.llm_model,
            "estimated_savings": None,
            "savings_percent": None,
            "message": (
                "Cost estimation is disabled for the unified Groq model."
            ),
        }


model_router = ModelRouter()
