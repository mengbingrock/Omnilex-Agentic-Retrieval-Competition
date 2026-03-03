"""Neo4j schema definitions for the Swiss legal citation graph.

Node labels
-----------
:Case       – A BGE Federal Court decision (one node per case, e.g. "BGE 139 I 2").
:Law        – A law provision cited by cases (e.g. "Art. 34 Abs. 1 BV").

Relationship types
------------------
(:Case)-[:CITES]->(:Case)         – Case cites another case.
(:Case)-[:CITES_LAW]->(:Law)      – Case cites a law provision.
"""

from neo4j import GraphDatabase

from .config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


SCHEMA_STATEMENTS: list[str] = [
    # ---- constraints ----
    "CREATE CONSTRAINT case_id IF NOT EXISTS FOR (c:Case) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT law_id IF NOT EXISTS FOR (l:Law) REQUIRE l.id IS UNIQUE",
    # ---- indexes for common lookups ----
    "CREATE INDEX case_volume IF NOT EXISTS FOR (c:Case) ON (c.volume)",
    "CREATE INDEX case_section IF NOT EXISTS FOR (c:Case) ON (c.section)",
    "CREATE INDEX law_book IF NOT EXISTS FOR (l:Law) ON (l.book)",
]


def create_schema(driver=None) -> list[str]:
    """Create constraints and indexes.  Returns list of executed statement summaries."""
    close = driver is None
    if close:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    executed: list[str] = []
    try:
        with driver.session() as session:
            for stmt in SCHEMA_STATEMENTS:
                try:
                    session.run(stmt)
                    executed.append(stmt.strip().split("\n")[0].strip())
                except Exception as exc:
                    if "already exists" not in str(exc).lower():
                        print(f"Warning: {exc}")
    finally:
        if close:
            driver.close()
    return executed


def drop_schema(driver=None) -> list[str]:
    """Drop all constraints and indexes."""
    close = driver is None
    if close:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    dropped: list[str] = []
    try:
        with driver.session() as session:
            for record in session.run("SHOW CONSTRAINTS"):
                name = record["name"]
                session.run(f"DROP CONSTRAINT {name} IF EXISTS")
                dropped.append(f"constraint:{name}")
            for record in session.run("SHOW INDEXES"):
                name = record["name"]
                if not name.startswith("constraint"):
                    session.run(f"DROP INDEX {name} IF EXISTS")
                    dropped.append(f"index:{name}")
    finally:
        if close:
            driver.close()
    return dropped


def clear_all_data(driver=None) -> int:
    """Delete every node and relationship (batched to avoid OOM)."""
    close = driver is None
    if close:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    total = 0
    try:
        with driver.session() as session:
            while True:
                result = session.run(
                    "MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(*) AS d"
                )
                deleted = result.single()["d"]
                total += deleted
                if deleted == 0:
                    break
    finally:
        if close:
            driver.close()
    return total


def get_schema_info(driver=None) -> dict:
    """Return current constraints and indexes as a dict."""
    close = driver is None
    if close:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        with driver.session() as session:
            constraints = [
                {
                    "name": r["name"],
                    "type": r["type"],
                    "entity_type": r["entityType"],
                    "labels": r["labelsOrTypes"],
                    "properties": r["properties"],
                }
                for r in session.run("SHOW CONSTRAINTS")
            ]
            indexes = [
                {
                    "name": r["name"],
                    "type": r["type"],
                    "entity_type": r["entityType"],
                    "labels": r["labelsOrTypes"],
                    "properties": r["properties"],
                    "state": r["state"],
                }
                for r in session.run("SHOW INDEXES")
            ]
            return {"constraints": constraints, "indexes": indexes}
    finally:
        if close:
            driver.close()


if __name__ == "__main__":
    import sys

    cmds = {"create": create_schema, "drop": drop_schema, "clear": clear_all_data, "info": get_schema_info}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print(f"Usage: python -m omnilex.graph.schema [{' | '.join(cmds)}]")
        raise SystemExit(1)
    result = cmds[sys.argv[1]]()
    print(result)
