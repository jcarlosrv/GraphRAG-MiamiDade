"""
Phase 1 — Structural load for Miami-Dade GraphRAG.

Reads Program-Descriptions-Miami-Dade.xlsx and builds the structural backbone in Neo4j:

    (Sector)-[:HAS_DEPT]->(Department)-[:HAS_PROGRAM]->(Program)

Key facts about the data (verified 2026-06-21):
  - 293 rows, 9 Sectors, 52 distinct Department *names*, 240 distinct Program *names*.
  - 9 Department names appear under >1 Sector   -> Department keyed by (Sector, Department).
  - 21 Program names appear under >1 Department -> Program keyed by (Sector, Department, Program).
  - 0 duplicate (Sector, Department, Program) triples -> exactly 293 Program nodes expected.
  - 'Key Activities' column is empty everywhere -> ignored.

Idempotent: re-running MERGEs on stable composite `key` properties, so no duplicates accrue.
"""

import os
import sys
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv(os.environ.get("GRAPHRAG_ENV_PATH"))

XLSX = "Department Narratives/Program-Descriptions-Miami-Dade.xlsx"
SHEET = "Program Descriptions"
SEP = " > "  # separator used inside composite keys

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
if not NEO4J_PASSWORD:
    print("ERROR: NEO4J_PASSWORD not set. Copy .env.example to .env and fill it in.", file=sys.stderr)
    sys.exit(2)

CONSTRAINTS = [
    "CREATE CONSTRAINT sector_key   IF NOT EXISTS FOR (s:Sector)     REQUIRE s.key IS UNIQUE",
    "CREATE CONSTRAINT dept_key     IF NOT EXISTS FOR (d:Department) REQUIRE d.key IS UNIQUE",
    "CREATE CONSTRAINT program_key  IF NOT EXISTS FOR (p:Program)    REQUIRE p.key IS UNIQUE",
]

LOAD_CYPHER = """
UNWIND $rows AS row
MERGE (s:Sector {key: row.sector})
  ON CREATE SET s.name = row.sector
MERGE (d:Department {key: row.dept_key})
  ON CREATE SET d.name = row.department, d.sector = row.sector
MERGE (s)-[:HAS_DEPT]->(d)
MERGE (p:Program {key: row.prog_key})
  ON CREATE SET p.name = row.program
  SET p.sector = row.sector,
      p.department = row.department,
      p.description = row.description,
      p.source = row.source
MERGE (d)-[:HAS_PROGRAM]->(p)
"""


def build_rows(df: pd.DataFrame) -> list[dict]:
    df = df.fillna("")
    rows = []
    for _, r in df.iterrows():
        sector = str(r["Sector"]).strip()
        department = str(r["Department"]).strip()
        program = str(r["Program"]).strip()
        dept_key = f"{sector}{SEP}{department}"
        prog_key = f"{sector}{SEP}{department}{SEP}{program}"
        rows.append(
            {
                "sector": sector,
                "department": department,
                "program": program,
                "dept_key": dept_key,
                "prog_key": prog_key,
                "description": str(r["Description"]).strip(),
                "source": str(r["Source"]).strip(),
            }
        )
    return rows


def main() -> int:
    df = pd.read_excel(XLSX, sheet_name=SHEET)
    rows = build_rows(df)
    print(f"Read {len(rows)} rows from {XLSX}")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()

    with driver.session() as session:
        for c in CONSTRAINTS:
            session.run(c)
        print("Constraints ensured.")

        session.run(LOAD_CYPHER, rows=rows)
        print("Load complete (MERGE, idempotent).")

        # Verify
        counts = session.run(
            """
            CALL () { MATCH (s:Sector)     RETURN count(s) AS sectors }
            CALL () { MATCH (d:Department) RETURN count(d) AS departments }
            CALL () { MATCH (p:Program)    RETURN count(p) AS programs }
            CALL () { MATCH (:Sector)-[r:HAS_DEPT]->()     RETURN count(r) AS has_dept }
            CALL () { MATCH (:Department)-[r:HAS_PROGRAM]->() RETURN count(r) AS has_program }
            RETURN sectors, departments, programs, has_dept, has_program
            """
        ).single()
        print("\n=== Graph contents ===")
        print(f"  Sectors:            {counts['sectors']}")
        print(f"  Departments:        {counts['departments']}")
        print(f"  Programs:           {counts['programs']}")
        print(f"  HAS_DEPT edges:     {counts['has_dept']}")
        print(f"  HAS_PROGRAM edges:  {counts['has_program']}")

    driver.close()

    expected_programs = len(rows)
    ok = counts["sectors"] == 9 and counts["programs"] == expected_programs
    print("\nOK" if ok else "\nWARNING: counts differ from expected")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
