#!/usr/bin/env python3
"""Ingest the DEx catalog into ChromaDB vector collections."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.catalog import Catalog
from src.embeddings import VectorStore

import config


def main():
    print(f"Loading catalog from {config.CATALOG_PATH}...")
    catalog = Catalog()
    print(f"  Found {len(catalog.products)} products, {len(catalog.relationships)} relationships")

    print(f"\nEmbedding and indexing into ChromaDB ({config.CHROMA_PERSIST_DIR})...")
    store = VectorStore()
    store.ingest_catalog(catalog)

    print("\nVerifying indices...")
    test_query = "What are the active bonds?"
    products = store.search_products(test_query, top_k=3)
    print(f"  Product search for '{test_query}':")
    for p in products:
        print(f"    - {p['metadata']['name']} (distance: {p['distance']:.4f})")

    examples = store.search_examples(test_query, top_k=2)
    print(f"  Example search for '{test_query}':")
    for e in examples:
        print(f"    - [{e['metadata']['product_name']}] {e['document']}")

    print("\nIngestion complete.")


if __name__ == "__main__":
    main()
