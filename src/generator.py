from __future__ import annotations

from src.catalog import Catalog, DataProduct
from src.embeddings import VectorStore
from src.llm import LLMProvider
from src.validator import validate_sql, validate_graphql

import config


SQL_SYSTEM_PROMPT = """You are an expert SQL query generator for Snowflake. Given a data product schema, business rules, and example queries, generate accurate Snowflake SQL for the user's question.

Rules:
- Use Snowflake SQL dialect (DATEADD, DATE_TRUNC, ILIKE, CURRENT_DATE(), etc.)
- Only reference columns that exist in the provided schema
- Apply relevant business rules automatically (e.g., filter active records, exclude cancelled items)
- Use meaningful aliases for readability
- Always wrap the SQL query in ```sql ... ``` code fences
- After the query, provide a brief explanation of what it does and any assumptions made

Schema:
{schema}

Business Rules:
{rules}

{examples_section}

{relationships_section}"""

GRAPHQL_SYSTEM_PROMPT = """You are an expert GraphQL query generator. Given a GraphQL schema (SDL), business rules, and example queries, generate accurate GraphQL queries for the user's question.

Rules:
- Generate valid GraphQL that conforms to the provided schema
- Use appropriate filters and arguments
- Request only the fields needed to answer the question
- Always wrap the GraphQL query in ```graphql ... ``` code fences
- After the query, provide a brief explanation of what it does

Schema (SDL):
{schema}

Business Rules:
{rules}

{examples_section}"""


class QueryGenerator:
    def __init__(
        self,
        catalog: Catalog,
        vector_store: VectorStore,
        llm: LLMProvider,
    ):
        self.catalog = catalog
        self.store = vector_store
        self.llm = llm

    def generate(self, question: str, product_id: str, query_type: str = "sql") -> dict:
        product = self.catalog.get_product(product_id)
        if not product:
            return {"error": f"Product '{product_id}' not found"}

        schema = product.get_schema_for_type(query_type)
        if not schema:
            return {"error": f"No {query_type} schema available for {product.name}"}

        rules_text = self._format_rules(product)
        examples_text = self._get_examples_section(question, product_id, query_type)
        relationships_text, related_products = self._get_relationships_section(
            question, product_id, query_type
        )

        if query_type == "graphql":
            system_prompt = GRAPHQL_SYSTEM_PROMPT.format(
                schema=schema,
                rules=rules_text,
                examples_section=examples_text,
            )
        else:
            system_prompt = SQL_SYSTEM_PROMPT.format(
                schema=schema,
                rules=rules_text,
                examples_section=examples_text,
                relationships_section=relationships_text,
            )

        last_error = None
        for attempt in range(config.MAX_VALIDATION_RETRIES + 1):
            user_prompt = question
            if last_error:
                user_prompt = (
                    f"{question}\n\n"
                    f"NOTE: Your previous query had errors: {last_error}\n"
                    f"Please fix these errors and try again."
                )

            response_text = self.llm.generate(system_prompt, user_prompt)
            parsed = self.llm.parse_query_response(response_text)

            if not parsed["query"]:
                last_error = "No query found in response"
                continue

            if query_type == "graphql":
                validation = validate_graphql(parsed["query"], product)
            else:
                validation = validate_sql(parsed["query"], product, extra_products=related_products)

            if validation["valid"]:
                return {
                    "query": parsed["query"],
                    "query_type": query_type,
                    "explanation": parsed["explanation"],
                    "product_id": product_id,
                    "product_name": product.name,
                    "validation": validation,
                    "attempts": attempt + 1,
                }

            last_error = "; ".join(validation["errors"])

        return {
            "query": parsed["query"],
            "query_type": query_type,
            "explanation": parsed["explanation"],
            "product_id": product_id,
            "product_name": product.name,
            "validation": {"valid": False, "errors": [last_error or "Validation failed"]},
            "attempts": config.MAX_VALIDATION_RETRIES + 1,
        }

    def _format_rules(self, product: DataProduct) -> str:
        if not product.business_rules:
            return "No specific business rules defined."
        lines = []
        for r in product.business_rules:
            lines.append(f"- {r.rule_name}: {r.description}")
            lines.append(f"  SQL: WHERE {r.sql_predicate}")
        return "\n".join(lines)

    def _get_examples_section(self, question: str, product_id: str, query_type: str) -> str:
        examples = self.store.search_examples(question, product_id=product_id)
        if not examples:
            return ""

        lines = ["Similar query examples:"]
        for ex in examples:
            meta = ex.get("metadata", {})
            if meta.get("query_type") == query_type:
                lines.append(f"\nQ: {ex['document']}")
                lines.append(f"A:\n```{query_type}\n{meta['query']}\n```")

        return "\n".join(lines) if len(lines) > 1 else ""

    def _get_relationships_section(
        self, question: str, product_id: str, query_type: str = "sql"
    ) -> tuple[str, list]:
        """Build relationships section with relevance-aware context injection.

        Uses embedding similarity + keyword matching to decide which related products
        get full context (schema + rules + value dictionaries) vs a minimal mention.

        Returns (prompt_text, list_of_related_DataProduct_objects that got full context).
        """
        related = self.catalog.get_related_products(product_id)
        if not related:
            return "", []

        unique_product_ids = list({r["product_id"] for r in related})
        relevance_scores = self.store.score_product_relevance(question, unique_product_ids)
        keyword_hits = self._check_keyword_relevance(question, unique_product_ids)

        relevant_ids = set()
        for pid in unique_product_ids:
            embedding_score = relevance_scores.get(pid, 0)
            has_keyword_hit = pid in keyword_hits
            if embedding_score >= config.RELATIONSHIP_RELEVANCE_THRESHOLD or has_keyword_hit:
                relevant_ids.add(pid)

        lines_relevant = []
        lines_other = []
        related_products = []
        seen = set()

        for r in related:
            other = self.catalog.get_product(r["product_id"])
            if not other:
                continue

            join_line = (
                f"JOIN {other.name} ({r['product_id']}): "
                f"ON {product_id}.{r['join_from']} = {r['product_id']}.{r['join_to']} "
                f"-- {r['description']}"
            )

            if r["product_id"] in relevant_ids:
                if r["product_id"] not in seen:
                    seen.add(r["product_id"])
                    related_products.append(other)

                    score = relevance_scores.get(r["product_id"], 0)
                    kw = keyword_hits.get(r["product_id"], [])
                    reason = f"relevance={score:.2f}"
                    if kw:
                        reason += f", keyword matches: {', '.join(kw[:5])}"

                    block = [f"\n- {join_line}"]
                    block.append(f"  (Included because: {reason})")

                    other_schema = other.get_schema_for_type(query_type)
                    if other_schema:
                        block.append(f"  Schema for {other.name}:\n  {other_schema}")

                    if other.business_rules:
                        block.append(f"  Business rules for {other.name}:")
                        for rule in other.business_rules:
                            block.append(f"    - {rule.rule_name}: {rule.description}")
                            block.append(f"      SQL: WHERE {rule.sql_predicate}")

                    value_dict_lines = []
                    for attr in other.attributes:
                        if attr.value_dictionary:
                            vals = ", ".join(
                                f"{k}={v}" for k, v in list(attr.value_dictionary.items())[:8]
                            )
                            value_dict_lines.append(f"    - {attr.physical_name}: {vals}")
                    if value_dict_lines:
                        block.append(f"  Key value lookups for {other.name}:")
                        block.extend(value_dict_lines)

                    lines_relevant.append("\n".join(block))
                else:
                    lines_relevant.append(f"\n- {join_line}")
            else:
                lines_other.append(f"  - {other.name} via {r['join_from']}")

        output = []
        if lines_relevant:
            output.append(
                "Related products (full context for question-relevant JOINs):"
            )
            output.extend(lines_relevant)
        if lines_other:
            output.append(
                "\nOther available JOINs (schema not included — use only if needed):"
            )
            output.extend(lines_other)

        return "\n".join(output), related_products

    def _check_keyword_relevance(
        self, question: str, product_ids: list[str]
    ) -> dict[str, list[str]]:
        """Check if any terms in the question match attribute names/synonyms in each product.

        Returns {product_id: [matched_terms]} for products with hits.
        """
        q_lower = question.lower()
        q_terms = set(q_lower.split())

        hits: dict[str, list[str]] = {}
        for pid in product_ids:
            product = self.catalog.get_product(pid)
            if not product:
                continue
            matched = []
            for attr in product.attributes:
                if attr.logical_name.lower() in q_lower:
                    matched.append(attr.logical_name)
                for syn in attr.synonyms:
                    if syn.lower() in q_lower:
                        matched.append(syn)
                if attr.value_dictionary:
                    for val_desc in attr.value_dictionary.values():
                        for term in q_terms:
                            if len(term) > 3 and term in val_desc.lower():
                                matched.append(f"{attr.physical_name}:{val_desc}")
                                break
            if matched:
                hits[pid] = list(dict.fromkeys(matched))
        return hits
