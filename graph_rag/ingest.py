"""
Graph RAG ingestion — run this once before querying.

Parses a Python directory, embeds all entities, and writes them into
Neo4j as a property graph (CodeEntity nodes + CALLS edges).

Usage:
    python -m graph_rag.ingest --path sample_codebase
    python -m graph_rag.ingest --path sample_codebase --clear
"""

import argparse
import logging
import sys

from graph_rag.code_parser import embed_entities, parse_directory
from graph_rag.graph_store import GraphStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Graph RAG — ingest Python source into Neo4j"
    )
    parser.add_argument(
        "--path", required=True, help="Directory of Python source files to ingest"
    )
    parser.add_argument(
        "--clear", action="store_true", help="Clear existing graph before ingesting"
    )
    args = parser.parse_args()

    logger.info("Step 1/3  Parsing: %s", args.path)
    entities = parse_directory(args.path)
    if not entities:
        logger.error("No entities found. Check the path and run from the project root.")
        sys.exit(1)
    logger.info("  -> %d entities extracted", len(entities))

    logger.info("Step 2/3  Embedding with sentence-transformers ...")
    entities = embed_entities(entities)
    logger.info("  -> embeddings ready (dim=384)")

    logger.info("Step 3/3  Writing to Neo4j ...")
    with GraphStore() as store:
        store.create_schema()
        if args.clear:
            logger.info("  -> clearing existing graph data")
            store.clear()
        store.build_graph(entities)

    calls_total = sum(len(e.calls) for e in entities)
    logger.info("Done: %d nodes, %d potential CALLS edges", len(entities), calls_total)
    logger.info("Neo4j Browser: http://localhost:7474  (neo4j / codebrain2024)")


if __name__ == "__main__":
    main()
