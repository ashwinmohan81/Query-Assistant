#!/usr/bin/env python3
"""Evaluate the Text-to-SQL pipeline against the golden dataset."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.catalog import Catalog
from src.embeddings import VectorStore
from src.llm import get_provider
from src.router import ProductRouter
from src.generator import QueryGenerator


def main():
    golden_path = Path(__file__).parent / "golden.json"
    golden = json.loads(golden_path.read_text())

    print("Initializing components...")
    catalog = Catalog()
    store = VectorStore()
    llm = get_provider()
    router = ProductRouter(catalog, store, llm)
    generator = QueryGenerator(catalog, store, llm)

    results = {
        "total": 0,
        "routing_correct": 0,
        "syntax_valid": 0,
        "schema_valid": 0,
        "details": [],
    }

    test_cases = golden["test_cases"]
    results["total"] = len(test_cases)

    print(f"\nRunning {len(test_cases)} test cases...\n")
    print(f"{'#':<4} {'Question':<55} {'Route':^8} {'Syntax':^8} {'Schema':^8}")
    print("-" * 87)

    for i, tc in enumerate(test_cases):
        question = tc["question"]
        expected_product = tc["expected_product"]

        start = time.time()

        routed = router.route(question)
        actual_product = routed[0]["product_id"] if routed else None
        query_type = routed[0].get("query_type", "sql") if routed else "sql"

        routing_ok = actual_product == expected_product
        if routing_ok:
            results["routing_correct"] += 1

        gen_result = generator.generate(question, actual_product or expected_product, query_type)
        validation = gen_result.get("validation", {})
        syntax_ok = validation.get("valid", False)
        if syntax_ok:
            results["syntax_valid"] += 1

        schema_ok = not any("not found" in e.lower() for e in validation.get("errors", []))
        if schema_ok:
            results["schema_valid"] += 1

        elapsed = time.time() - start

        route_sym = "PASS" if routing_ok else "FAIL"
        syntax_sym = "PASS" if syntax_ok else "FAIL"
        schema_sym = "PASS" if schema_ok else "FAIL"

        short_q = question[:52] + "..." if len(question) > 55 else question
        print(f"{i+1:<4} {short_q:<55} {route_sym:^8} {syntax_sym:^8} {schema_sym:^8}")

        results["details"].append({
            "question": question,
            "expected_product": expected_product,
            "actual_product": actual_product,
            "routing_correct": routing_ok,
            "syntax_valid": syntax_ok,
            "schema_valid": schema_ok,
            "query": gen_result.get("query", ""),
            "errors": validation.get("errors", []),
            "elapsed_s": round(elapsed, 2),
        })

    print("\n" + "=" * 87)
    total = results["total"]
    print(f"\nResults ({total} test cases):")
    print(f"  Routing accuracy:  {results['routing_correct']}/{total} ({results['routing_correct']/total*100:.0f}%)")
    print(f"  Syntax validity:   {results['syntax_valid']}/{total} ({results['syntax_valid']/total*100:.0f}%)")
    print(f"  Schema fidelity:   {results['schema_valid']}/{total} ({results['schema_valid']/total*100:.0f}%)")

    output_path = Path(__file__).parent / "results.json"
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()
