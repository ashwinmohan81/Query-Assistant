from __future__ import annotations

import re

import sqlglot
from sqlglot import exp

from src.catalog import DataProduct


def _extract_ddl_columns(ddl: str | None) -> set[str]:
    """Extract column names from a CREATE TABLE DDL statement."""
    if not ddl:
        return set()
    cols = set()
    for match in re.finditer(r'^\s+(\w+)\s+\w+', ddl, re.MULTILINE):
        name = match.group(1).lower()
        if name.upper() not in ('CREATE', 'TABLE', 'PRIMARY', 'KEY', 'FOREIGN', 'CONSTRAINT', 'UNIQUE', 'INDEX'):
            cols.add(name)
    return cols


def validate_sql(query: str, product: DataProduct, extra_products: list[DataProduct] | None = None) -> dict:
    """Validate SQL query against the product schema (and any joined products)."""
    if not query.strip():
        return {"valid": False, "errors": ["Empty query"]}

    errors = []

    dangerous = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "TRUNCATE", "CREATE", "GRANT"]
    upper_query = query.upper().strip()
    for kw in dangerous:
        if upper_query.startswith(kw):
            errors.append(f"Dangerous operation not allowed: {kw}")
            return {"valid": False, "errors": errors}

    try:
        parsed = sqlglot.parse(query, read="snowflake")
    except sqlglot.errors.ParseError as e:
        errors.append(f"SQL syntax error: {e}")
        return {"valid": False, "errors": errors}

    if not parsed:
        errors.append("Could not parse query")
        return {"valid": False, "errors": errors}

    all_products = [product] + (extra_products or [])
    schema_columns: set[str] = set()
    for p in all_products:
        for a in p.attributes:
            schema_columns.add(a.physical_name.lower())
        schema_columns.update(_extract_ddl_columns(p.schema_sql))

    for statement in parsed:
        if statement is None:
            continue
        for col in statement.find_all(exp.Column):
            col_name = col.name.lower()
            if col_name not in schema_columns and col_name != "*":
                errors.append(f"Column '{col.name}' not found in schema of any referenced product")

    if errors:
        return {"valid": False, "errors": errors}

    return {"valid": True, "errors": []}


def validate_graphql(query: str, product: DataProduct) -> dict:
    """Basic GraphQL validation against the product SDL schema."""
    if not query.strip():
        return {"valid": False, "errors": ["Empty query"]}

    if not product.schema_graphql:
        return {"valid": False, "errors": ["No GraphQL schema available for this product"]}

    errors = []

    try:
        from graphql import parse as gql_parse, build_schema, validate as gql_validate

        schema = build_schema(product.schema_graphql)
        document = gql_parse(query)
        gql_errors = gql_validate(schema, document)
        for err in gql_errors:
            errors.append(f"GraphQL validation error: {err.message}")
    except Exception as e:
        errors.append(f"GraphQL parse error: {e}")

    if errors:
        return {"valid": False, "errors": errors}

    return {"valid": True, "errors": []}
