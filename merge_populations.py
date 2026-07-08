"""
merge_populations.py — normalize near-duplicate :Population entities into canonical nodes.

Idempotent + re-runnable. For each (canonical -> [variants]) group:
  - ensures the canonical :Entity:Population node exists (keyed by normalized name),
  - re-points every MENTIONS and RELATES_TO edge from each variant to the canonical
    (deduped via MERGE; self-loops onto the canonical are dropped),
  - absorbs the variant's surface form into the canonical's `aliases` (deduped),
  - DETACH DELETEs the variant.

Entity identity follows the extractor: key = norm(name) = whitespace-collapsed, trimmed, lowercased.
Run from the venv:  .venv/Scripts/python.exe merge_populations.py
"""
import os
import re
import sys
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv(r"C:\Users\jcarl\Quant16\Others\GraphRAG\.env")

PWD = os.environ.get("NEO4J_PASSWORD")
if not PWD:
    print("ERROR: NEO4J_PASSWORD not set. Copy .env.example to .env and fill it in.", file=sys.stderr)
    sys.exit(2)
URI, USER = os.environ.get("NEO4J_URI", "bolt://localhost:7687"), os.environ.get("NEO4J_USER", "neo4j")

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()

# canonical display name -> variant surface forms it absorbs
GROUPS = {
    "residents": ["Miami-Dade County residents", "public", "taxpayers", "community"],
    "residents with disabilities": ["people with varied abilities",
                                    "residents and visitors with disabilities"],
    "county employees": ["countywide employees", "County workforce"],
    "vendors": ["vendor community", "supplier community", "bidders"],
    "community-based organizations": ["not-for-profit organizations"],
    "Internal Compliance staff": ["Internal Compliance Assistant",
                                  "Internal Compliance Associates",
                                  "Internal Compliance Senior"],
    "County departments": ["County agencies", "user departments"],
    "businesses": ["business community", "small businesses"],
}

ENSURE_CANONICAL = """
MERGE (c:Entity {key: $ckey})
  ON CREATE SET c.name = $cname, c.aliases = []
SET c:Population
"""

REPOINT_MENTIONS = """
MATCH (v)-[:MENTIONS]->(d:Entity {key: $vkey})
MATCH (c:Entity {key: $ckey})
MERGE (v)-[:MENTIONS]->(c)
"""

REPOINT_RELATES_OUT = """
MATCH (d:Entity {key: $vkey})-[r:RELATES_TO]->(o)
MATCH (c:Entity {key: $ckey})
WHERE o.key <> $ckey
MERGE (c)-[r2:RELATES_TO {type: r.type}]->(o)
  ON CREATE SET r2.evidence = r.evidence
"""

REPOINT_RELATES_IN = """
MATCH (o)-[r:RELATES_TO]->(d:Entity {key: $vkey})
MATCH (c:Entity {key: $ckey})
WHERE o.key <> $ckey
MERGE (o)-[r2:RELATES_TO {type: r.type}]->(c)
  ON CREATE SET r2.evidence = r.evidence
"""

ABSORB_AND_DELETE = """
MATCH (c:Entity {key: $ckey}), (d:Entity {key: $vkey})
WITH c, d, coalesce(c.aliases, []) + coalesce(d.aliases, []) + [d.name] AS al
UNWIND al AS a
WITH c, d, collect(DISTINCT a) AS uniq
SET c.aliases = [x IN uniq WHERE x IS NOT NULL AND x <> c.name]
DETACH DELETE d
"""

COUNT = "MATCH (p:Population) RETURN count(p) AS n"


def main():
    drv = GraphDatabase.driver(URI, auth=(USER, PWD))
    with drv.session() as s:
        before = s.run(COUNT).single()["n"]
        print(f"Population nodes before: {before}\n")

        for canonical, variants in GROUPS.items():
            ckey = norm(canonical)
            s.run(ENSURE_CANONICAL, ckey=ckey, cname=canonical)
            merged = []
            for v in variants:
                vkey = norm(v)
                if vkey == ckey:
                    continue
                # only act if the variant still exists
                exists = s.run("MATCH (d:Entity {key:$vkey}) RETURN count(d) AS n",
                               vkey=vkey).single()["n"]
                if not exists:
                    continue
                s.run(REPOINT_MENTIONS, ckey=ckey, vkey=vkey)
                s.run(REPOINT_RELATES_OUT, ckey=ckey, vkey=vkey)
                s.run(REPOINT_RELATES_IN, ckey=ckey, vkey=vkey)
                s.run(ABSORB_AND_DELETE, ckey=ckey, vkey=vkey)
                merged.append(v)
            tag = f"  <- {', '.join(merged)}" if merged else "  (already canonical / nothing to merge)"
            print(f"  {canonical}{tag}")

        after = s.run(COUNT).single()["n"]
        print(f"\nPopulation nodes after: {after}  (removed {before - after})")
    drv.close()


if __name__ == "__main__":
    main()
