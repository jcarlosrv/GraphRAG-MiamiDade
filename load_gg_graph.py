"""
Load the new General Government model into Neo4j (fresh rebuild).

Reads General_Government.xlsx (built by build_gg_workbook.py) and loads:

    (Sector)-[:HAS_DEPARTMENT]->(Department)-[:HAS_DIVISION]->(Division)

Replaces the old xlsx-derived Sector/Department/Program graph entirely:
this script DETACH DELETEs all existing nodes first (fresh rebuild, confirmed).

Node properties
  Sector      : key, name
  Department  : key, name, description, total_fte, fiscal_year,
                unattributed_budget_fy24_25, unattributed_budget_fy25_26, source_pdf
  Division    : key, name, description, services (list), num_services,
                comments, fte_fy24_25, fte_fy25_26,
                budget_fy24_25, budget_fy25_26, source_pdf

Keys are composite + " > "-separated, matching the Phase-1 convention, so the
load is idempotent via MERGE.
"""

import os
import sys
import math
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv(os.environ.get("GRAPHRAG_ENV_PATH"))

XLSX = "General_Government.xlsx"
SECTOR = "General Government"
SEP = " > "

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
if not NEO4J_PASSWORD:
    print("ERROR: NEO4J_PASSWORD not set. Copy .env.example to .env and fill it in.", file=sys.stderr)
    sys.exit(2)

CONSTRAINTS = [
    "CREATE CONSTRAINT sector_key   IF NOT EXISTS FOR (s:Sector)     REQUIRE s.key IS UNIQUE",
    "CREATE CONSTRAINT dept_key     IF NOT EXISTS FOR (d:Department) REQUIRE d.key IS UNIQUE",
    "CREATE CONSTRAINT division_key IF NOT EXISTS FOR (v:Division)   REQUIRE v.key IS UNIQUE",
]

LOAD_DEPTS = """
MERGE (s:Sector {key: $sector})
  ON CREATE SET s.name = $sector
WITH s
UNWIND $rows AS row
MERGE (d:Department {key: row.key})
  SET d.name = row.name,
      d.description = row.description,
      d.total_fte = row.total_fte,
      d.fiscal_year = row.fiscal_year,
      d.unattributed_budget_fy24_25 = row.unattributed_budget_fy24_25,
      d.unattributed_budget_fy25_26 = row.unattributed_budget_fy25_26,
      d.source_pdf = row.source_pdf
MERGE (s)-[:HAS_DEPARTMENT]->(d)
"""

LOAD_DIVS = """
UNWIND $rows AS row
MATCH (d:Department {key: row.dept_key})
MERGE (v:Division {key: row.key})
  SET v.name = row.name,
      v.description = row.description,
      v.services = row.services,
      v.num_services = row.num_services,
      v.comments = row.comments,
      v.fte_fy24_25 = row.fte_fy24_25,
      v.fte_fy25_26 = row.fte_fy25_26,
      v.budget_fy24_25 = row.budget_fy24_25,
      v.budget_fy25_26 = row.budget_fy25_26,
      v.source_pdf = row.source_pdf
MERGE (d)-[:HAS_DIVISION]->(v)
"""


def _int(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return int(x)


def _str(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return str(x)


def build_payload():
    dep_df = pd.read_excel(XLSX, sheet_name="Departments")
    div_df = pd.read_excel(XLSX, sheet_name="Divisions")

    dept_rows = []
    for _, r in dep_df.iterrows():
        name = _str(r["department"]).strip()
        dept_rows.append({
            "key": f"{SECTOR}{SEP}{name}",
            "name": name,
            "description": _str(r["description"]),
            "total_fte": _int(r["total_fte"]),
            "fiscal_year": _str(r["fiscal_year"]),
            "unattributed_budget_fy24_25": _int(r["unattributed_budget_fy24_25"]),
            "unattributed_budget_fy25_26": _int(r["unattributed_budget_fy25_26"]),
            "source_pdf": _str(r["source_pdf"]),
        })

    div_rows = []
    for _, r in div_df.iterrows():
        dept = _str(r["department"]).strip()
        name = _str(r["division"]).strip()
        services = [s.strip() for s in _str(r["services"]).split(" | ") if s.strip()]
        comments = [c.strip() for c in _str(r["comments"]).split(" | ") if c.strip()]
        div_rows.append({
            "dept_key": f"{SECTOR}{SEP}{dept}",
            "key": f"{SECTOR}{SEP}{dept}{SEP}{name}",
            "name": name,
            "description": _str(r["division_description"]),
            "services": services,
            "num_services": len(services),
            "comments": comments,
            "fte_fy24_25": _int(r["fte_fy24_25"]),
            "fte_fy25_26": _int(r["fte_fy25_26"]),
            "budget_fy24_25": _int(r["budget_fy24_25"]),
            "budget_fy25_26": _int(r["budget_fy25_26"]),
            "source_pdf": _str(r["source_pdf"]),
        })

    return dept_rows, div_rows


def main() -> int:
    dept_rows, div_rows = build_payload()
    print(f"Read {len(dept_rows)} departments, {len(div_rows)} divisions from {XLSX}")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()

    with driver.session() as s:
        before = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        s.run("MATCH (n) DETACH DELETE n")
        print(f"Cleared existing graph ({before} nodes removed).")

        for c in CONSTRAINTS:
            s.run(c)
        print("Constraints ensured.")

        s.run(LOAD_DEPTS, sector=SECTOR, rows=dept_rows)
        s.run(LOAD_DIVS, rows=div_rows)
        print("Load complete (MERGE, idempotent).")

        counts = s.run(
            """
            CALL () { MATCH (s:Sector)      RETURN count(s) AS sectors }
            CALL () { MATCH (d:Department)  RETURN count(d) AS departments }
            CALL () { MATCH (v:Division)    RETURN count(v) AS divisions }
            CALL () { MATCH (:Sector)-[r:HAS_DEPARTMENT]->()    RETURN count(r) AS has_dept }
            CALL () { MATCH (:Department)-[r:HAS_DIVISION]->()  RETURN count(r) AS has_div }
            CALL () { MATCH (v:Division) WHERE v.fte_fy25_26 IS NOT NULL RETURN count(v) AS with_fte }
            RETURN sectors, departments, divisions, has_dept, has_div, with_fte
            """
        ).single()
        print("\n=== Graph contents ===")
        print(f"  Sectors:               {counts['sectors']}")
        print(f"  Departments:           {counts['departments']}")
        print(f"  Divisions:             {counts['divisions']}")
        print(f"  HAS_DEPARTMENT edges:  {counts['has_dept']}")
        print(f"  HAS_DIVISION edges:    {counts['has_div']}")
        print(f"  Divisions with FTE:    {counts['with_fte']}/{counts['divisions']}")

    driver.close()
    ok = (counts["sectors"] == 1 and counts["departments"] == len(dept_rows)
          and counts["divisions"] == len(div_rows))
    print("\nOK" if ok else "\nWARNING: counts differ from expected")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
