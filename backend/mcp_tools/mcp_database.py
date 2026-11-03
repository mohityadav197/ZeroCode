"""Read-only, bounded PostgreSQL queries (also supports SQLite for tests)."""
import json

import sqlglot
from sqlglot import exp
from sqlalchemy import text

from database import engine


def query_database(sql: str) -> dict:
    """Execute one read-only SELECT and return at most 1000 JSON records."""
    statements = sqlglot.parse(sql, read="postgres")
    if len(statements) != 1 or not isinstance(statements[0], (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        raise ValueError("Only a single read-only SELECT is supported")
    tree = statements[0]
    forbidden = (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Alter, exp.Into, exp.Lock, exp.Command)
    if any(isinstance(node, forbidden) for node in tree.walk()):
        raise ValueError("Mutating queries are blocked")
    if any(isinstance(fn, exp.Anonymous) for fn in tree.find_all(exp.Func)):
        raise ValueError("Unknown or side-effecting database functions are blocked")
    if any(fn.sql_name().lower() in {"pg_sleep", "set_config", "dblink_exec", "lo_import", "lo_export"}
           or getattr(fn, "name", "").lower() in {"pg_sleep", "set_config", "dblink_exec", "lo_import", "lo_export"}
           for fn in tree.find_all(exp.Func)):
        raise ValueError("Side-effecting functions are blocked")
    dialect = "postgres" if engine.dialect.name == "postgresql" else "sqlite"
    bounded = f"SELECT * FROM ({tree.sql(dialect=dialect)}) AS zerocode_readonly LIMIT 1000"
    with engine.connect() as connection, connection.begin():
        if engine.dialect.name == "postgresql":
            connection.exec_driver_sql("SET TRANSACTION READ ONLY")
            connection.exec_driver_sql("SET LOCAL statement_timeout = '5000ms'")
        result = connection.execute(text(bounded))
        columns = list(result.keys())
        rows = [dict(row) for row in result.mappings()]
    rows = json.loads(json.dumps(rows, default=str))
    return {"columns": columns, "rows": rows, "row_count": len(rows)}
