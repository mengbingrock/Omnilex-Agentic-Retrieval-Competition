"""Neo4j local service management for the Omnilex citation graph.

Manages a Homebrew-installed Neo4j instance (``brew install neo4j``).
"""

import subprocess
import time

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from .config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


def _brew(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["neo4j", *args], capture_output=True, text=True, check=check
    )


def is_container_running() -> bool:
    """Check if the local Neo4j service is running."""
    r = subprocess.run(
        ["neo4j", "status"], capture_output=True, text=True, check=False
    )
    return "is running" in r.stdout.lower() or "is running" in r.stderr.lower()


# Alias kept for backward-compat with __init__.py
is_running = is_container_running


def check_health() -> bool:
    """True if Neo4j accepts bolt connections."""
    try:
        drv = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with drv.session() as s:
            s.run("RETURN 1").single()
        drv.close()
        return True
    except (ServiceUnavailable, Exception):
        return False


def wait_for_ready(timeout: int = 60) -> bool:
    print("Waiting for Neo4j to be ready...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if check_health():
            print("Neo4j is ready!")
            return True
        time.sleep(2)
    print(f"Timeout after {timeout}s waiting for Neo4j.")
    return False


def start_neo4j(wait: bool = True, timeout: int = 60, **_kwargs) -> bool:
    """Start the local Neo4j service (Homebrew install).

    Equivalent to ``neo4j start``.
    """
    if is_container_running():
        print("Neo4j is already running.")
        return True

    print("Starting Neo4j (local)...")
    r = subprocess.run(
        ["neo4j", "start"], capture_output=True, text=True, check=False
    )
    if r.returncode != 0:
        print(f"neo4j start failed: {r.stderr.strip() or r.stdout.strip()}")
        return False

    return wait_for_ready(timeout) if wait else True


def stop_neo4j() -> bool:
    """Stop the local Neo4j service."""
    if not is_container_running():
        print("Neo4j is not running.")
        return True
    print("Stopping Neo4j...")
    r = subprocess.run(
        ["neo4j", "stop"], capture_output=True, text=True, check=False
    )
    return r.returncode == 0


def restart_neo4j() -> bool:
    """Restart Neo4j."""
    stop_neo4j()
    return start_neo4j()


def get_connection():
    """Return a Neo4j driver instance."""
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def get_stats() -> dict:
    """Quick node / relationship counts."""
    drv = get_connection()
    try:
        with drv.session() as s:
            nodes = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            rels = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            cases = s.run("MATCH (n:Case) RETURN count(n) AS c").single()["c"]
            laws = s.run("MATCH (n:Law) RETURN count(n) AS c").single()["c"]
            cites = s.run("MATCH ()-[r:CITES]->() RETURN count(r) AS c").single()["c"]
            cites_law = s.run("MATCH ()-[r:CITES_LAW]->() RETURN count(r) AS c").single()["c"]
        return {
            "total_nodes": nodes,
            "total_relationships": rels,
            "cases": cases,
            "laws": laws,
            "cites_edges": cites,
            "cites_law_edges": cites_law,
        }
    finally:
        drv.close()


if __name__ == "__main__":
    import sys

    usage = "Usage: python -m omnilex.graph.docker_setup [start|stop|restart|status]"
    if len(sys.argv) < 2:
        print(usage)
        raise SystemExit(1)

    cmd = sys.argv[1].lower()
    if cmd == "start":
        start_neo4j()
    elif cmd == "stop":
        stop_neo4j()
    elif cmd == "restart":
        restart_neo4j()
    elif cmd == "status":
        if is_container_running():
            print("Neo4j is running.")
            if check_health():
                for k, v in get_stats().items():
                    print(f"  {k}: {v}")
            else:
                print("  (not healthy yet)")
        else:
            print("Neo4j is not running.")
    else:
        print(f"Unknown: {cmd}\n{usage}")
        raise SystemExit(1)
