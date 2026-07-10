"""
Phase 2 — LLM entity + relationship extraction over the General Government graph.

For each Division already loaded by load_gg_graph.py, send its
description + services + comments to Claude (structured output) and extract
typed entities and within-division relationships per the fixed 6-type taxonomy,
then MERGE the results into Neo4j:

    (Division)-[:MENTIONS]->(Entity)
    (Entity)-[:RELATES_TO {type, evidence}]->(Entity)

Entities carry the shared :Entity label plus a per-type label
(:Entity:Service, :Entity:Regulation, ...). Identity is the canonical
lowercased `name`, used as the MERGE key, so "Florida Statutes" / "Florida
statutes" collapse to ONE node linked to every Division that cites it — the
basis for cross-division GraphRAG queries. The original surface form(s) are
kept in `aliases`.

Resumable + idempotent: each Division is flagged `extracted = true` on success
and skipped on re-run; all writes are MERGE.

Run:
    ANTHROPIC_API_KEY in .env
    # ensure Neo4j is running, then:
    python extract_gg_entities.py            # extract all not-yet-done divisions
    python extract_gg_entities.py --force    # re-extract everything (clears flags)
"""

import os
import re
import sys
import argparse
from enum import Enum
from typing import List

import anthropic
from pydantic import BaseModel, Field
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv(os.environ.get("GRAPHRAG_ENV_PATH"))

MODEL = "claude-sonnet-4-6"  # extraction model
MAX_TOKENS = 8000

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
if not NEO4J_PASSWORD:
    print("ERROR: NEO4J_PASSWORD not set. Copy .env.example to .env and fill it in.", file=sys.stderr)
    sys.exit(2)

# ----------------------------------------------------------------------------
# Extraction schema — fixed 6-type taxonomy
# ----------------------------------------------------------------------------

class EntityType(str, Enum):
    Service = "Service"        # an action/activity the government performs
    Population = "Population"  # who is served / affected
    Location = "Location"      # places, facilities, geographic areas
    Agency = "Agency"          # orgs/depts/external bodies named in text
    Asset = "Asset"            # physical/financial things managed
    Regulation = "Regulation"  # laws, policies, standards cited


class Entity(BaseModel):
    name: str = Field(description="The entity as it appears in the text (surface form).")
    type: EntityType = Field(description="One of the six fixed entity types.")


class Relationship(BaseModel):
    source: str = Field(description="Exact `name` of the source entity (must match an entity above).")
    target: str = Field(description="Exact `name` of the target entity (must match an entity above).")
    type: str = Field(description="Short verb phrase, e.g. complies_with, serves, manages, operates.")
    evidence: str = Field(description="The source snippet stating this relationship.")


class Extraction(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]


SYSTEM_PROMPT = """\
You extract a knowledge graph from Miami-Dade County government program text.

Extract TYPED ENTITIES using EXACTLY these six types — never invent others:
- Service: an action/activity the government performs (e.g. "invests surplus funds", "permit issuance")
- Population: who is served or affected (e.g. "residents", "seniors", "small businesses")
- Location: places, facilities, geographic areas (e.g. "Miami-Dade County", "wastewater plants")
- Agency: organizations / departments / external bodies NAMED in the text (e.g. "Florida Dept. of Revenue", "the Board")
- Asset: physical or financial things managed (e.g. "surplus funds", "fleet vehicles", "the 911 system")
- Regulation: laws, policies, standards cited (e.g. "Florida Statutes", "local ordinances", "investment policy")

Also extract RELATIONSHIPS between the entities you found. Rules:
- Only relationships STATED WITHIN this text. Do NOT infer or add outside knowledge.
- `type` is a short snake_case verb phrase (complies_with, serves, manages, operates, funds, ...).
- `evidence` is the exact source snippet that states the relationship.
- `source` and `target` must each exactly match the `name` of an entity you listed.

Be precise and conservative. If something is not clearly one of the six types, omit it."""


def build_input(div: dict) -> str:
    parts = [f"DEPARTMENT: {div['dept']}", f"DIVISION: {div['name']}", ""]
    if div.get("description"):
        parts += ["DESCRIPTION:", div["description"], ""]
    if div.get("services"):
        parts += ["SERVICES:"] + [f"- {s}" for s in div["services"]] + [""]
    if div.get("comments"):
        parts += ["COMMENTS:"] + [f"- {c}" for c in div["comments"]] + [""]
    return "\n".join(parts).strip()


# ----------------------------------------------------------------------------
# Neo4j
# ----------------------------------------------------------------------------

CONSTRAINT = "CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE"

# One MERGE per entity type so the per-type label can be a literal (Cypher
# can't parameterize labels). `key` is the canonical name → cross-division merge.
MERGE_ENTITIES_TMPL = """
UNWIND $rows AS row
MERGE (e:Entity {{key: row.key}})
  ON CREATE SET e.name = row.name
  SET e:{label},
      e.type = $etype,
      e.aliases = CASE
        WHEN row.mention IN coalesce(e.aliases, []) THEN e.aliases
        ELSE coalesce(e.aliases, []) + row.mention
      END
WITH e, row
MATCH (v:Division {{key: row.div_key}})
MERGE (v)-[:MENTIONS]->(e)
"""

MERGE_RELS = """
UNWIND $rows AS row
MATCH (a:Entity {key: row.source_key})
MATCH (b:Entity {key: row.target_key})
MERGE (a)-[r:RELATES_TO {type: row.type}]->(b)
  SET r.evidence = row.evidence
"""

MARK_DONE = """
MATCH (v:Division {key: $key})
SET v.extracted = true, v.entity_count = $n
"""


def norm(name: str) -> str:
    """Canonical identity: trim + collapse whitespace + lowercase."""
    return re.sub(r"\s+", " ", name).strip().lower()


def fetch_divisions(session, force: bool) -> list:
    if force:
        session.run("MATCH (v:Division) REMOVE v.extracted, v.entity_count")
        session.run("MATCH (e:Entity) DETACH DELETE e")
    rows = session.run(
        """
        MATCH (d:Department)-[:HAS_DIVISION]->(v:Division)
        WHERE coalesce(v.extracted, false) = false
        RETURN v.key AS key, v.name AS name, d.name AS dept,
               v.description AS description, v.services AS services,
               v.comments AS comments
        ORDER BY v.key
        """
    )
    return [dict(r) for r in rows]


def write_extraction(session, div_key: str, ex: Extraction) -> int:
    # Normalize + de-dupe entities by canonical key, grouped by type.
    key_for = {}
    by_type = {}
    for e in ex.entities:
        key = norm(e.name)
        if not key:
            continue
        key_for[e.name] = key  # map surface form -> key for relationship resolution
        by_type.setdefault(e.type.value, {})[key] = e.name  # keep one surface form

    for etype, ents in by_type.items():
        rows = [{"key": k, "name": surf, "mention": surf, "div_key": div_key}
                for k, surf in ents.items()]
        session.run(MERGE_ENTITIES_TMPL.format(label=etype), rows=rows, etype=etype)

    # Resolve relationship endpoints to canonical keys; drop dangling edges.
    valid_keys = {k for ents in by_type.values() for k in ents}
    rel_rows = []
    for r in ex.relationships:
        sk, tk = norm(r.source), norm(r.target)
        if sk in valid_keys and tk in valid_keys and sk != tk and r.type.strip():
            rel_rows.append({"source_key": sk, "target_key": tk,
                             "type": r.type.strip(), "evidence": r.evidence})
    if rel_rows:
        session.run(MERGE_RELS, rows=rel_rows)

    n = len(valid_keys)
    session.run(MARK_DONE, key=div_key, n=n)
    return n


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Clear extracted flags + all Entity nodes, re-extract from scratch.")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set. (Phase 2 only.)\n"
              '  setx ANTHROPIC_API_KEY "..."  then restart the shell.', file=sys.stderr)
        return 2

    client = anthropic.Anthropic()
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
    except Exception as e:
        print(f"ERROR: cannot reach Neo4j at {NEO4J_URI} ({e}).\n"
              "  Start Neo4j, then re-run.", file=sys.stderr)
        return 3

    with driver.session() as session:
        session.run(CONSTRAINT)
        divisions = fetch_divisions(session, args.force)
        total = len(divisions)
        if total == 0:
            print("Nothing to extract — all divisions already flagged extracted. "
                  "(Use --force to redo.)")
            return 0
        print(f"Extracting {total} division(s) with {MODEL}...\n")

        ok = 0
        for i, div in enumerate(divisions, 1):
            try:
                resp = client.messages.parse(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": build_input(div)}],
                    output_format=Extraction,
                )
                ex = resp.parsed_output
                if ex is None:  # e.g. refusal / truncation
                    print(f"  [{i}/{total}] {div['name']:40} SKIPPED "
                          f"(stop_reason={resp.stop_reason})")
                    continue
                n = write_extraction(session, div["key"], ex)
                ok += 1
                print(f"  [{i}/{total}] {div['name']:40} "
                      f"entities={n:3} rels={len(ex.relationships):3}")
            except anthropic.APIError as e:
                # Leave the flag unset so a re-run resumes this division.
                print(f"  [{i}/{total}] {div['name']:40} API ERROR: {e}", file=sys.stderr)

        counts = session.run(
            """
            CALL () { MATCH (e:Entity) RETURN count(e) AS entities }
            CALL () { MATCH (:Division)-[m:MENTIONS]->() RETURN count(m) AS mentions }
            CALL () { MATCH (:Entity)-[r:RELATES_TO]->() RETURN count(r) AS relates }
            CALL () { MATCH (v:Division) WHERE v.extracted RETURN count(v) AS done }
            RETURN entities, mentions, relates, done
            """
        ).single()

    driver.close()
    print("\n=== Phase 2 results ===")
    print(f"  Divisions extracted this run: {ok}/{total}")
    print(f"  Total divisions flagged done: {counts['done']}")
    print(f"  Entity nodes:                 {counts['entities']}")
    print(f"  MENTIONS edges:               {counts['mentions']}")
    print(f"  RELATES_TO edges:             {counts['relates']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
