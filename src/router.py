from __future__ import annotations

from src.catalog import Catalog, DataProduct
from src.embeddings import VectorStore
from src.llm import LLMProvider

import config


RERANK_PROMPT = """You are a data product routing assistant. Given a user's natural language question and a list of candidate data products, determine which product(s) are most relevant.

Candidate products:
{candidates}

User question: {question}

Respond in this exact format:
PRODUCT: <product_id>
CONFIDENCE: <0.0 to 1.0>
REASONING: <one sentence>
QUERY_TYPE: <sql or graphql>

If the question requires joining data from multiple products, list each on a separate line with the same format. Only include products that are genuinely needed."""


class ProductRouter:
    def __init__(self, catalog: Catalog, vector_store: VectorStore, llm: LLMProvider | None = None):
        self.catalog = catalog
        self.store = vector_store
        self.llm = llm

    def route(self, question: str) -> list[dict]:
        """Route a question to the most relevant data product(s)."""
        candidates = self.store.search_products(question, top_k=config.TOP_K_PRODUCTS)

        if not candidates:
            return []

        if self.llm and not isinstance(self.llm, type) and config.LLM_PROVIDER != "mock":
            return self._rerank_with_llm(question, candidates)

        return self._rank_by_vector_score(candidates)

    def _rank_by_vector_score(self, candidates: list[dict]) -> list[dict]:
        """Fallback: rank purely by embedding similarity."""
        results = []
        for c in candidates:
            product_id = c["metadata"]["product_id"]
            product = self.catalog.get_product(product_id)
            if not product:
                continue

            confidence = max(0.0, 1.0 - c["distance"])
            query_type = "graphql" if product.distribution_type == "radix" else "sql"
            if product.distribution_type == "both":
                query_type = "sql"

            results.append({
                "product_id": product_id,
                "product_name": product.name,
                "confidence": round(confidence, 3),
                "query_type": query_type,
                "reasoning": f"Matched by semantic similarity (score: {confidence:.3f})",
            })

        return results

    def _rerank_with_llm(self, question: str, candidates: list[dict]) -> list[dict]:
        """Use LLM to rerank and select the best product(s)."""
        candidate_text = ""
        for c in candidates:
            product_id = c["metadata"]["product_id"]
            product = self.catalog.get_product(product_id)
            if not product:
                continue
            dist_types = []
            if product.schema_sql:
                dist_types.append("SQL/Snowflake")
            if product.schema_graphql:
                dist_types.append("GraphQL/Radix")
            candidate_text += (
                f"\n- ID: {product_id}\n"
                f"  Name: {product.name}\n"
                f"  Domain: {product.domain} > {product.subdomain}\n"
                f"  Description: {product.description}\n"
                f"  Available as: {', '.join(dist_types)}\n"
                f"  Key attributes: {', '.join(a.logical_name for a in product.attributes[:10])}\n"
            )

        prompt = RERANK_PROMPT.format(candidates=candidate_text, question=question)

        try:
            response = self.llm.generate(
                system_prompt="You are a precise data product routing assistant. Always respond in the exact format requested.",
                user_prompt=prompt,
            )
            return self._parse_rerank_response(response, candidates)
        except Exception as e:
            print(f"LLM reranking failed ({e}), falling back to vector scores")
            return self._rank_by_vector_score(candidates)

    def _parse_rerank_response(self, response: str, candidates: list[dict]) -> list[dict]:
        """Parse the structured LLM reranking response."""
        results = []
        current = {}

        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("PRODUCT:"):
                if current:
                    results.append(current)
                current = {"product_id": line.split(":", 1)[1].strip()}
            elif line.startswith("CONFIDENCE:"):
                try:
                    current["confidence"] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    current["confidence"] = 0.5
            elif line.startswith("REASONING:"):
                current["reasoning"] = line.split(":", 1)[1].strip()
            elif line.startswith("QUERY_TYPE:"):
                current["query_type"] = line.split(":", 1)[1].strip().lower()

        if current and "product_id" in current:
            results.append(current)

        valid_ids = {c["metadata"]["product_id"] for c in candidates}
        validated = []
        for r in results:
            pid = r.get("product_id", "")
            if pid in valid_ids:
                product = self.catalog.get_product(pid)
                r.setdefault("product_name", product.name if product else pid)
                r.setdefault("confidence", 0.5)
                r.setdefault("query_type", "sql")
                r.setdefault("reasoning", "Selected by LLM reranking")
                validated.append(r)

        if not validated:
            return self._rank_by_vector_score(candidates)

        return sorted(validated, key=lambda x: x["confidence"], reverse=True)
