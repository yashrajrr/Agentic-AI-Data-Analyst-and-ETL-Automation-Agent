from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import re
import json
import time
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
import psycopg2
from dotenv import dotenv_values

_venv_dotenv = dotenv_values(Path(__file__).resolve().parent / ".env")
for _k, _v in _venv_dotenv.items():
    os.environ.setdefault(_k, _v or "")

from langchain_core.messages import HumanMessage, ToolMessage
from agents.data_agent import data_agent
from utils.database import DatabaseUtil

app = FastAPI(title="Data Agent API")

# Allow CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "index.html"
DATA_DIR = BASE_DIR / "data"
HISTORY_PATH = DATA_DIR / "history.json"
HISTORY_LIMIT = 50
ALLOWED_FOLDERS = {"extract", "transform"}



# ----------------------------- DATABASE HELPERS ----------------------------- #

def get_db_config():
    return {
        "host": os.environ.get("host", "localhost"),
        "port": int(os.environ.get("port", 5432)),
        "user": os.environ.get("user"),
        "password": os.environ.get("password"),
        "dbname": os.environ.get("database"),
    }


def structured_query(sql):
    """Run a query and return a JSON-safe tabular result (never raises)."""
    empty = {"columns": [], "rows": [], "row_count": 0, "error": None}
    if not sql:
        return {**empty, "error": "no query to execute"}
    try:
        obj = DatabaseUtil(get_db_config())
        return obj.execute_sql_structured(sql)
    except Exception as e:
        return {**empty, "error": str(e)}


# ------------------------------ HISTORY STORE ------------------------------- #

def load_history():
    if not HISTORY_PATH.exists():
        return []
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_history(items):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = HISTORY_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(items[:HISTORY_LIMIT], f, ensure_ascii=False, indent=2)
    tmp.replace(HISTORY_PATH)


def append_history(question, route, answer):
    items = load_history()
    items.insert(0, {
        "id": f"{int(time.time() * 1000)}",
        "question": question,
        "route": route,
        "answer": answer or "",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    save_history(items)
    return items


# --------------------------- AGENT OUTPUT PARSING --------------------------- #

def message_content(message):
    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", None)


def message_tool_calls(message):
    if isinstance(message, dict):
        return message.get("tool_calls") or []
    return getattr(message, "tool_calls", None) or []


def is_tool_message(message):
    if isinstance(message, ToolMessage):
        return True
    if isinstance(message, dict):
        return message.get("type") == "tool"
    return type(message).__name__ == "ToolMessage"


def last_ai_text(messages):
    """Return the last non-empty AI message that did not request a tool call."""
    for message in reversed(messages or []):
        content = message_content(message)
        if isinstance(content, str) and content.strip() and not message_tool_calls(message):
            return content
    return ""


def extract_sql_payload(state):
    """Build the structured ``sql`` block from the SQL agent's final state."""
    generated_sql = state.get("generated_sql_query") or ""
    is_safe = state.get("is_safe") or ""
    comments = state.get("comments") or ""
    answer = state.get("final_answer") or ""

    table: dict[str, object] = {"columns": [], "rows": [], "row_count": 0, "error": None}
    if generated_sql and is_safe == "Yes":
        table = structured_query(generated_sql)

    if not answer:
        answer = last_ai_text(state.get("messages"))

    # Normalize tooling wrappers into plain dicts so the payload is JSON-safe.
    return {
        "query": generated_sql,
        "is_safe": is_safe,
        "comments": comments,
        "columns": table["columns"],
        "rows": table["rows"],
        "row_count": table["row_count"],
        "error": table["error"],
        "raw_result": state.get("sql_query_execution_result", "") or "",
        "curated_question": state.get("curated_ques", "") or "",
    }, answer


def parse_output_path(text):
    for pattern in (r"saved at ([^\n]+?) in ", r"saved to ([^\n]+)"):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def parse_generated_code(text):
    match = re.search(r"Pandas Code Executed:(.*?)(?:Execution Result:|$)", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_etl_payload(inner_messages):
    """Build the structured ``etl`` block from the ETL agent's message list.

    Walks the ETL agent's messages to recover the tool output and the final
    answer, then parses the output path and generated pandas code from the
    tool output text. Tools any surrounding text so that ``content`` may be
    ``None`` or contain the tool output as substring.
    """
    tool_outputs: list[str] = []
    for message in inner_messages or []:
        content = message_content(message)
        text = content if isinstance(content, str) else ""
        if not text.strip():
            continue
        if is_tool_message(message):
            # Some tool agents wrap the tool output with surrounding prose;
            # keep the whole text so the path/code parsers can find their
            # substrings.
            tool_outputs.append(text)

    tool_output_text = "\n".join(tool_outputs)
    tool_output = tool_output_text
    if tool_output_text and "Pandas Code Executed" in tool_output_text:
        tool = "transform_load"
    elif "extracted and saved" in tool_output or "Failed to extract" in tool_output:
        tool = "extract_load"
    else:
        tool = ""

    answer = last_ai_text(inner_messages) or tool_output_text

    return {
        "message": tool_output_text,
        "tool": tool,
        "output_path": parse_output_path(tool_output_text),
        "code": parse_generated_code(tool_output_text),
        "success": bool(tool_output_text) and "Fail" not in tool_output_text[:60],
    }, answer


# -------------------------------- ENDPOINTS --------------------------------- #

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/db-info")
async def get_db_info():
    try:
        conn = psycopg2.connect(
            host=os.environ.get("host", "localhost"),
            port=int(os.environ.get("port", 5432)),
            database=os.environ.get("database"),
            user=os.environ.get("user"),
            password=os.environ.get("password"),
        )
        cursor = conn.cursor()

        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
        """)
        tables = [row[0] for row in cursor.fetchall()]

        db_metadata = []
        for table in tables:
            cursor.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
            """, (table,))
            columns = [row[0] for row in cursor.fetchall()]
            db_metadata.append({"table": table, "columns": columns})

        cursor.close()
        conn.close()

        return {
            "status": "success",
            "connected": True,
            "server": "PostgreSQL",
            "database": os.environ.get("database"),
            "tables": db_metadata,
        }
    except Exception as e:
        return {
            "status": "error",
            "connected": False,
            "message": str(e),
        }


@app.post("/api/ask")
async def ask_question(request: Request):
    data = await request.json()
    query = (data.get("query") or "").strip()

    if not query:
        return {"status": "error", "message": "No query provided"}

    started = time.perf_counter()

    try:
        response = data_agent.invoke(
            {"messages": [HumanMessage(content=query)], "route_response": ""}
        )
    except Exception as e:
        # Never surface raw agent internals or stack traces to the client.
        msg = str(e)
        if len(msg) > 600:
            msg = msg[:600].rstrip() + "…"
        return {"status": "error", "message": msg}

    route = response.get("route_response", "unknown")
    messages = response.get("messages", [])
    last = messages[-1] if messages else None

    payload = {
        "status": "success",
        "route": route,
        "answer": "",
        "sql": None,
        "etl": None,
        "timing_ms": 0,
    }

    if route == "sql" and isinstance(last, dict):
        sql_payload, answer = extract_sql_payload(last)
        payload["sql"] = sql_payload
        payload["answer"] = answer or last_ai_text(messages)
    elif route == "etl" and isinstance(last, dict):
        etl_payload, answer = extract_etl_payload(last.get("messages"))
        payload["etl"] = etl_payload
        payload["answer"] = answer or etl_payload["message"]
    else:
        payload["answer"] = last_ai_text(messages) or str(last)

    payload["timing_ms"] = round((time.perf_counter() - started) * 1000)

    try:
        append_history(query, route, payload["answer"])
    except Exception:
        # Never let history write break the user's answer.
        pass

    return payload


@app.get("/api/history")
async def get_history():
    return {"status": "success", "history": load_history()}


@app.post("/api/history")
async def post_history(request: Request):
    data = await request.json()
    question = (data.get("question") or "").strip()
    if not question:
        return {"status": "error", "message": "No question provided"}
    if not question:
        return {"status": "error", "message": "No question provided"}
    items = append_history(question, data.get("route", "unknown"), data.get("answer", ""))
    return {"status": "success", "history": items}


@app.get("/api/files")
async def get_files():
    files = []
    for folder in sorted(ALLOWED_FOLDERS):
        folder_path = DATA_DIR / folder
        if not folder_path.is_dir():
            continue
        for entry in folder_path.iterdir():
            if not entry.is_file() or entry.name.startswith("."):
                continue
            stat = entry.stat()
            files.append({
                "folder": folder,
                "name": entry.name,
                "path": f"{folder}/{entry.name}",
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
            })
    files.sort(key=lambda f: f["modified"], reverse=True)
    return {"status": "success", "files": files}


@app.get("/api/files/{folder}/{name}")
async def download_file(folder: str, name: str):
    if folder not in ALLOWED_FOLDERS:
        raise HTTPException(status_code=404, detail="Unknown folder")
    if Path(name).name != name or name.startswith("."):
        raise HTTPException(status_code=404, detail="Invalid file name")

    target = (DATA_DIR / folder / name).resolve()
    if DATA_DIR.resolve() not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    media_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    return FileResponse(target, media_type=media_type, filename=target.name)


@app.get("/images/{name}")
async def serve_image(name: str):
    if name in ["data_agent_graph.png", "sql_analyst_graph.png", "etl_analyst_graph.png"]:
        path = BASE_DIR / name
        if path.exists():
            return FileResponse(path)
    raise HTTPException(status_code=404, detail="Image not found")


@app.get("/api/metadata")
async def get_metadata():
    """Return column-level metadata + row counts for every public table."""
    try:
        conn = psycopg2.connect(
            host=os.environ.get("host", "localhost"),
            port=int(os.environ.get("port", 5432)),
            database=os.environ.get("database"),
            user=os.environ.get("user"),
            password=os.environ.get("password"),
        )
        cur = conn.cursor()

        # All tables in public schema
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        tables = [r[0] for r in cur.fetchall()]

        result = []
        for table in tables:
            # Column details
            cur.execute("""
                SELECT
                    column_name,
                    data_type,
                    is_nullable,
                    column_default,
                    character_maximum_length
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
                ORDER BY ordinal_position
            """, (table,))
            columns = [
                {
                    "name": row[0],
                    "type": row[1],
                    "nullable": row[2] == "YES",
                    "default": row[3],
                    "max_length": row[4],
                }
                for row in cur.fetchall()
            ]

            # Row count
            cur.execute(
                f'SELECT COUNT(*) FROM public."{table}"'  # noqa: S608
            )
            row_count = cur.fetchone()[0]

            result.append({
                "table": table,
                "row_count": row_count,
                "columns": columns,
            })

        cur.close()
        conn.close()
        return {"status": "success", "tables": result}

    except Exception as exc:
        return {"status": "error", "message": str(exc), "tables": []}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)

