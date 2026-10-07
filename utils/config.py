import os
from urllib.parse import parse_qsl, unquote, urlparse


def get_db_config() -> dict[str, object]:
    """Return a psycopg2 config from Supabase/Vercel or local env vars."""
    database_url = os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")
    if database_url:
        parsed = urlparse(database_url)
        query = dict(parse_qsl(parsed.query))
        config: dict[str, object] = {
            "host": parsed.hostname,
            "port": parsed.port or 5432,
            "user": unquote(parsed.username or ""),
            "password": unquote(parsed.password or ""),
            "dbname": unquote(parsed.path.lstrip("/")),
        }
        if "sslmode" in query:
            config["sslmode"] = query["sslmode"]
        elif "supabase" in (parsed.hostname or ""):
            config["sslmode"] = "require"
        return config

    database = os.environ.get("POSTGRES_DB") or os.environ.get("database")
    return {
        "host": os.environ.get("POSTGRES_HOST") or os.environ.get("host", "localhost"),
        "port": int(os.environ.get("POSTGRES_PORT") or os.environ.get("port", 5432)),
        "user": os.environ.get("POSTGRES_USER") or os.environ.get("user"),
        "password": os.environ.get("POSTGRES_PASSWORD") or os.environ.get("password"),
        "dbname": database,
    }


def get_display_database_name() -> str | None:
    config = get_db_config()
    value = config.get("dbname")
    return str(value) if value else None
