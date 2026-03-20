from __future__ import annotations

from sentence_transformers import SentenceTransformer
import chromadb

import config
from src.catalog import Catalog, DataProduct


_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def get_chroma_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)


class VectorStore:
    PRODUCTS_COLLECTION = "products"
    ATTRIBUTES_COLLECTION = "attributes"
    EXAMPLES_COLLECTION = "examples"

    def __init__(self):
        self.client = get_chroma_client()
        self.model = get_embedding_model()

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def ingest_catalog(self, catalog: Catalog):
        self._ingest_products(catalog)
        self._ingest_attributes(catalog)
        self._ingest_examples(catalog)

    def _ingest_products(self, catalog: Catalog):
        col = self.client.get_or_create_collection(
            self.PRODUCTS_COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        col.delete(where={"product_id": {"$ne": ""}})

        docs, ids, metas = [], [], []
        for p in catalog.list_products():
            text = p.build_embedding_text()
            docs.append(text)
            ids.append(p.product_id)
            metas.append({
                "product_id": p.product_id,
                "name": p.name,
                "domain": p.domain,
                "subdomain": p.subdomain,
                "distribution_type": p.distribution_type,
            })

        embeddings = self._embed(docs)
        col.add(documents=docs, embeddings=embeddings, ids=ids, metadatas=metas)
        print(f"  Indexed {len(docs)} products")

    def _ingest_attributes(self, catalog: Catalog):
        col = self.client.get_or_create_collection(
            self.ATTRIBUTES_COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        col.delete(where={"product_id": {"$ne": ""}})

        docs, ids, metas = [], [], []
        for p in catalog.list_products():
            for attr in p.attributes:
                synonyms_str = ", ".join(attr.synonyms) if attr.synonyms else ""
                values_str = ""
                if attr.value_dictionary:
                    values_str = " | Values: " + ", ".join(
                        f"{k} ({v})" for k, v in attr.value_dictionary.items()
                    )

                text = (
                    f"{attr.logical_name} ({attr.physical_name}): {attr.description}"
                    f"{' | Also known as: ' + synonyms_str if synonyms_str else ''}"
                    f"{values_str}"
                    f" | Product: {p.name} | Type: {attr.data_type}"
                )
                doc_id = f"{p.product_id}__{attr.physical_name}"
                docs.append(text)
                ids.append(doc_id)
                metas.append({
                    "product_id": p.product_id,
                    "product_name": p.name,
                    "physical_name": attr.physical_name,
                    "logical_name": attr.logical_name,
                    "data_type": attr.data_type,
                })

        embeddings = self._embed(docs)
        col.add(documents=docs, embeddings=embeddings, ids=ids, metadatas=metas)
        print(f"  Indexed {len(docs)} attributes")

    def _ingest_examples(self, catalog: Catalog):
        col = self.client.get_or_create_collection(
            self.EXAMPLES_COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        col.delete(where={"product_id": {"$ne": ""}})

        docs, ids, metas = [], [], []
        idx = 0
        for p in catalog.list_products():
            for ex in p.query_examples:
                docs.append(ex.question)
                ids.append(f"{p.product_id}__ex_{idx}")
                metas.append({
                    "product_id": p.product_id,
                    "product_name": p.name,
                    "query_type": ex.query_type,
                    "query": ex.query,
                })
                idx += 1

        embeddings = self._embed(docs)
        col.add(documents=docs, embeddings=embeddings, ids=ids, metadatas=metas)
        print(f"  Indexed {len(docs)} query examples")

    def search_products(self, query: str, top_k: int | None = None) -> list[dict]:
        top_k = top_k or config.TOP_K_PRODUCTS
        col = self.client.get_collection(self.PRODUCTS_COLLECTION)
        embedding = self._embed([query])[0]
        results = col.query(query_embeddings=[embedding], n_results=top_k)
        return self._format_results(results)

    def search_attributes(
        self, query: str, product_id: str | None = None, top_k: int | None = None
    ) -> list[dict]:
        top_k = top_k or config.TOP_K_ATTRIBUTES
        col = self.client.get_collection(self.ATTRIBUTES_COLLECTION)
        embedding = self._embed([query])[0]
        where = {"product_id": product_id} if product_id else None
        results = col.query(query_embeddings=[embedding], n_results=top_k, where=where)
        return self._format_results(results)

    def search_examples(
        self, query: str, product_id: str | None = None, top_k: int | None = None
    ) -> list[dict]:
        top_k = top_k or config.TOP_K_EXAMPLES
        col = self.client.get_collection(self.EXAMPLES_COLLECTION)
        embedding = self._embed([query])[0]
        where = {"product_id": product_id} if product_id else None
        results = col.query(query_embeddings=[embedding], n_results=top_k, where=where)
        return self._format_results(results)

    def score_product_relevance(self, query: str, product_ids: list[str]) -> dict[str, float]:
        """Score how relevant each product is to the query. Returns {product_id: similarity}."""
        if not product_ids:
            return {}
        query_embedding = self._embed([query])[0]
        col = self.client.get_collection(self.PRODUCTS_COLLECTION)
        results = col.get(ids=product_ids, include=["embeddings"])
        scores = {}
        if len(results["ids"]) > 0 and results["embeddings"] is not None:
            import numpy as np
            q = np.array(query_embedding)
            for i, pid in enumerate(results["ids"]):
                p = np.array(results["embeddings"][i])
                cosine_sim = float(np.dot(q, p) / (np.linalg.norm(q) * np.linalg.norm(p) + 1e-10))
                scores[pid] = cosine_sim
        return scores

    @staticmethod
    def _format_results(results: dict) -> list[dict]:
        formatted = []
        if not results["ids"] or not results["ids"][0]:
            return formatted
        for i, doc_id in enumerate(results["ids"][0]):
            item = {
                "id": doc_id,
                "document": results["documents"][0][i] if results["documents"] else "",
                "distance": results["distances"][0][i] if results["distances"] else 0,
            }
            if results["metadatas"] and results["metadatas"][0]:
                item["metadata"] = results["metadatas"][0][i]
            formatted.append(item)
        return formatted
