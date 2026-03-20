#!/usr/bin/env python3
"""FastAPI server for the Text-to-SQL/GraphQL prototype."""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.catalog import Catalog
from src.embeddings import VectorStore
from src.llm import get_provider
from src.router import ProductRouter
from src.generator import QueryGenerator

app = FastAPI(title="Text-to-SQL/GraphQL Prototype", version="0.1.0")
templates = Jinja2Templates(directory="templates")

catalog: Catalog | None = None
vector_store: VectorStore | None = None
router: ProductRouter | None = None
generator: QueryGenerator | None = None


def get_components():
    global catalog, vector_store, router, generator
    if catalog is None:
        catalog = Catalog()
        vector_store = VectorStore()
        llm = get_provider()
        router = ProductRouter(catalog, vector_store, llm)
        generator = QueryGenerator(catalog, vector_store, llm)
    return catalog, vector_store, router, generator


class QueryRequest(BaseModel):
    question: str
    product_id: str | None = None
    query_type: str | None = None


class QueryResponse(BaseModel):
    question: str
    routed_products: list[dict]
    selected_product: dict | None
    query: str
    query_type: str
    explanation: str
    validation: dict
    attempts: int


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/query")
async def generate_query(req: QueryRequest):
    cat, store, rtr, gen = get_components()

    if req.product_id:
        product = cat.get_product(req.product_id)
        if not product:
            return {"error": f"Product '{req.product_id}' not found"}
        routed = [{
            "product_id": req.product_id,
            "product_name": product.name,
            "confidence": 1.0,
            "query_type": req.query_type or "sql",
            "reasoning": "Manually selected by user",
        }]
    else:
        routed = rtr.route(req.question)

    if not routed:
        return {"error": "Could not identify a relevant data product for this question."}

    selected = routed[0]
    product_id = selected["product_id"]
    query_type = req.query_type or selected.get("query_type", "sql")

    result = gen.generate(req.question, product_id, query_type)

    return {
        "question": req.question,
        "routed_products": routed,
        "selected_product": selected,
        "query": result.get("query", ""),
        "query_type": result.get("query_type", query_type),
        "explanation": result.get("explanation", ""),
        "validation": result.get("validation", {}),
        "attempts": result.get("attempts", 0),
        "error": result.get("error"),
    }


@app.get("/api/products")
async def list_products():
    cat, *_ = get_components()
    return [
        {
            "product_id": p.product_id,
            "name": p.name,
            "domain": p.domain,
            "subdomain": p.subdomain,
            "distribution_type": p.distribution_type,
            "attribute_count": len(p.attributes),
        }
        for p in cat.list_products()
    ]


@app.get("/api/products/{product_id}")
async def get_product(product_id: str):
    cat, *_ = get_components()
    product = cat.get_product(product_id)
    if not product:
        return {"error": "Not found"}
    return {
        "product_id": product.product_id,
        "name": product.name,
        "domain": product.domain,
        "subdomain": product.subdomain,
        "description": product.description,
        "distribution_type": product.distribution_type,
        "attributes": [
            {"logical_name": a.logical_name, "physical_name": a.physical_name, "data_type": a.data_type}
            for a in product.attributes
        ],
        "business_rules": [
            {"rule_name": r.rule_name, "description": r.description}
            for r in product.business_rules
        ],
        "relationships": cat.get_related_products(product_id),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8090, reload=True)
