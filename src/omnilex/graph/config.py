"""Neo4j connection configuration for the Swiss legal citation graph."""

import os

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "omnilex123")

CONTAINER_NAME = "omnilex-neo4j"
NEO4J_IMAGE = "neo4j:5.15.0"
NEO4J_HTTP_PORT = 7474
NEO4J_BOLT_PORT = 7687
NEO4J_MEMORY = "2G"

BATCH_SIZE = 500
