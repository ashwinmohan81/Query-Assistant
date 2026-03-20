from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import config


@dataclass
class Attribute:
    logical_name: str
    physical_name: str
    data_type: str
    description: str
    synonyms: list[str] = field(default_factory=list)
    graphql_name: Optional[str] = None
    value_dictionary: Optional[dict[str, str]] = None
    foreign_key: Optional[str] = None
    is_primary_key: bool = False


@dataclass
class BusinessRule:
    rule_name: str
    description: str
    sql_predicate: str
    graphql_filter: Optional[str] = None


@dataclass
class QueryExample:
    question: str
    query_type: str
    query: str


@dataclass
class Relationship:
    from_product: str
    from_field: str
    to_product: str
    to_field: str
    description: str


@dataclass
class DataProduct:
    product_id: str
    name: str
    domain: str
    subdomain: str
    description: str
    distribution_type: str  # "datamesh" | "radix" | "both"
    schema_sql: Optional[str]
    schema_graphql: Optional[str]
    attributes: list[Attribute]
    business_rules: list[BusinessRule]
    query_examples: list[QueryExample]

    def get_attribute_names(self) -> list[str]:
        names = []
        for attr in self.attributes:
            names.append(attr.logical_name)
            names.extend(attr.synonyms)
        return names

    def get_schema_for_type(self, query_type: str) -> Optional[str]:
        if query_type == "graphql" and self.schema_graphql:
            return self.schema_graphql
        return self.schema_sql

    def build_embedding_text(self) -> str:
        attr_text = ", ".join(
            f"{a.logical_name} ({', '.join(a.synonyms[:3])})" if a.synonyms else a.logical_name
            for a in self.attributes[:20]
        )
        return (
            f"Product: {self.name}\n"
            f"Domain: {self.domain} > {self.subdomain}\n"
            f"Description: {self.description}\n"
            f"Key Attributes: {attr_text}"
        )


class Catalog:
    def __init__(self, path: Optional[str] = None):
        self.path = Path(path or config.CATALOG_PATH)
        self.products: dict[str, DataProduct] = {}
        self.relationships: list[Relationship] = []
        self._load()

    def _load(self):
        raw = json.loads(self.path.read_text())
        for p in raw["products"]:
            attrs = [Attribute(**a) for a in p["attributes"]]
            rules = [BusinessRule(**r) for r in p["business_rules"]]
            examples = [QueryExample(**e) for e in p["query_examples"]]
            product = DataProduct(
                product_id=p["product_id"],
                name=p["name"],
                domain=p["domain"],
                subdomain=p["subdomain"],
                description=p["description"],
                distribution_type=p["distribution_type"],
                schema_sql=p.get("schema_sql"),
                schema_graphql=p.get("schema_graphql"),
                attributes=attrs,
                business_rules=rules,
                query_examples=examples,
            )
            self.products[product.product_id] = product

        for r in raw.get("cross_product_relationships", []):
            self.relationships.append(Relationship(**r))

    def get_product(self, product_id: str) -> Optional[DataProduct]:
        return self.products.get(product_id)

    def list_products(self) -> list[DataProduct]:
        return list(self.products.values())

    def get_related_products(self, product_id: str) -> list[dict]:
        related = []
        for r in self.relationships:
            if r.from_product == product_id:
                related.append({
                    "product_id": r.to_product,
                    "join_from": r.from_field,
                    "join_to": r.to_field,
                    "description": r.description,
                })
            elif r.to_product == product_id:
                related.append({
                    "product_id": r.from_product,
                    "join_from": r.to_field,
                    "join_to": r.from_field,
                    "description": r.description,
                })
        return related
