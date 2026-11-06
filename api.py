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
from typing import Any
import psycopg2
from dotenv import dotenv_values

_venv_dotenv = dotenv_values(Path(__file__).resolve().parent / ".env")
for _k, _v in _venv_dotenv.items():
    os.environ.setdefault(_k, _v or "")

from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from agents.data_agent import data_agent
from utils.database import DatabaseUtil
from utils.config import get_db_config, get_display_database_name

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
ALLOWED_FOLDERS = {"extract", "transform", "uploads"}



# ----------------------------- DATABASE HELPERS ----------------------------- #

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

    table: dict[str, Any] = {"columns": [], "rows": [], "row_count": 0, "error": None}
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
    match = re.search(r"saved (?:at|to) (.+?\.(?:csv|json|parquet))\b", text, re.IGNORECASE)
    return match.group(1).strip() if match else ""


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
        conn = psycopg2.connect(**get_db_config())
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
            "database": get_display_database_name(),
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
    selected_dataset = (data.get("dataset") or "auto").strip()

    if not query:
        return {"status": "error", "message": "No query provided"}
    if selected_dataset != "auto":
        available = {item["value"] for item in (await get_datasets()).get("datasets", [])}
        if selected_dataset != "database" and selected_dataset not in available:
            return {"status": "error", "message": "The selected dataset is no longer available. Refresh the page and choose it again."}

    started = time.perf_counter()

    try:
        past = load_history()
        history_msgs = []
        # Keep the last 5 exchanges for context
        for item in reversed(past[:5]):
            history_msgs.append(HumanMessage(content=item["question"]))
            # Some answers might be empty or pure data, ensure string
            history_msgs.append(AIMessage(content=str(item["answer"] or "")))
            
        history_msgs.append(HumanMessage(content=query))

        response = data_agent.invoke(
            {"messages": history_msgs, "route_response": "", "selected_dataset": selected_dataset}
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

    payload: dict[str, Any] = {
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
    items = append_history(question, data.get("route", "unknown"), data.get("answer", ""))
    return {"status": "success", "history": items}


@app.delete("/api/history")
async def delete_history():
    save_history([])
    return {"status": "success", "history": []}


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


@app.get("/api/datasets")
async def get_datasets():
    """List selectable database tables and local data files for the query UI."""
    tables = []
    try:
        conn = psycopg2.connect(**get_db_config())
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                    ORDER BY table_name
                """)
                tables = [{"value": f"table:{row[0]}", "label": f"Database · {row[0]}", "type": "table"}
                          for row in cursor.fetchall()]
        finally:
            conn.close()
    except Exception:
        # The data-file selector remains useful when the SQL database is offline.
        pass

    files = []
    for folder in sorted(ALLOWED_FOLDERS):
        folder_path = DATA_DIR / folder
        if not folder_path.is_dir():
            continue
        for entry in folder_path.iterdir():
            if entry.is_file() and entry.suffix.lower() in {".csv", ".json", ".parquet"} and not entry.name.startswith("."):
                path = f"data/{folder}/{entry.name}"
                files.append({"value": f"file:{path}", "label": f"File · {path}", "type": "file"})
    files.sort(key=lambda item: item["label"].lower())
    return {"status": "success", "datasets": tables + files}
    
from fastapi import UploadFile, File

@app.post("/api/files/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        target_dir = DATA_DIR / "uploads"
        target_dir.mkdir(parents=True, exist_ok=True)
        safe_filename = Path(file.filename or "upload.tmp").name
        if safe_filename.startswith(".") or Path(safe_filename).suffix.lower() not in {".csv", ".json", ".parquet"}:
            raise HTTPException(status_code=400, detail="Upload a CSV, JSON, or Parquet file with a valid filename.")
        target_path = target_dir / safe_filename
        content = await file.read(25 * 1024 * 1024 + 1)
        if len(content) > 25 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File exceeds the 25 MB upload limit.")
        with open(target_path, "wb") as f:
            f.write(content)
        return {"status": "success", "path": f"data/uploads/{safe_filename}"}
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        return {"status": "error", "message": str(e)}


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
        conn = psycopg2.connect(**get_db_config())
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
            row = cur.fetchone()
            row_count = row[0] if row else 0

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


@app.post("/api/query-suggestions")
async def suggest_database_queries(request: Request):
    """Generate fresh natural-language questions grounded in database or file metadata."""
    data = await request.json()
    selected_dataset = (data.get("dataset") or "database").strip()
    previous_suggestions = data.get("exclude") or []
    if not isinstance(previous_suggestions, list):
        previous_suggestions = []
    selected_table = selected_dataset.removeprefix("table:") if selected_dataset.startswith("table:") else None
    try:
        if selected_dataset.startswith("file:"):
            relative_path = Path(selected_dataset.removeprefix("file:")).as_posix()
            parts = Path(relative_path).parts
            if len(parts) != 3 or parts[0] != "data" or parts[1] not in ALLOWED_FOLDERS or Path(parts[2]).name != parts[2] or Path(parts[2]).suffix.lower() not in {".csv", ".json", ".parquet"}:
                raise HTTPException(status_code=400, detail="The selected data file is invalid.")
            file_path = (BASE_DIR / relative_path).resolve()
            if DATA_DIR.resolve() not in file_path.parents or not file_path.is_file():
                raise HTTPException(status_code=404, detail="The selected data file is unavailable. Refresh the file list and try again.")
            import pandas as pd
            suffix = file_path.suffix.lower()
            if suffix == ".csv":
                frame = pd.read_csv(file_path, nrows=100)
            elif suffix == ".json":
                try:
                    frame = pd.read_json(file_path, lines=True, nrows=100)
                except ValueError:
                    frame = pd.read_json(file_path)
                    frame = frame.head(100)
            else:
                frame = pd.read_parquet(file_path).head(100)
            metadata = [{
                "table": file_path.name,
                "columns": [{"name": str(name), "type": str(dtype)} for name, dtype in frame.dtypes.items()],
            }]
        else:
            conn = psycopg2.connect(**get_db_config())
            try:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT table_name FROM information_schema.tables
                        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                        ORDER BY table_name
                    """)
                    table_names = [row[0] for row in cursor.fetchall()]
                    if selected_table:
                        if selected_table not in table_names:
                            raise HTTPException(status_code=400, detail="The selected database table is unavailable.")
                        table_names = [selected_table]

                    metadata = []
                    for table_name in table_names:
                        cursor.execute("""
                            SELECT column_name, data_type
                            FROM information_schema.columns
                            WHERE table_schema = 'public' AND table_name = %s
                            ORDER BY ordinal_position
                        """, (table_name,))
                        metadata.append({
                            "table": table_name,
                            "columns": [{"name": row[0], "type": row[1]} for row in cursor.fetchall()],
                        })
            finally:
                conn.close()

        if not metadata:
            return {"status": "success", "suggestions": [], "message": "No database tables are available."}

        table_context = "\n".join(
            f"- {table['table']}: " + ", ".join(f"{column['name']} ({column['type']})" for column in table["columns"])
            for table in metadata
        )
        from utils.llm_pick import pick_llm
        response = pick_llm("low").invoke(f"""
Generate exactly 3 distinct, useful natural-language questions a user can ask a read-only analytics assistant about this PostgreSQL database. Every refresh must produce questions different from the excluded suggestions below.
Ground every question in the table and column metadata below. Do not invent fields or tables. Prefer a mix of useful summaries, comparisons or trends, and at least one chart-worthy question when suitable date/time and numeric or categorical fields exist. For a single-table selection, use only that table. Questions must be concise and understandable to a non-technical user.
Return only a JSON array of 3 strings, with no markdown or additional text.

Previously shown questions to avoid:
{json.dumps(previous_suggestions[:30], ensure_ascii=False)}

Database metadata:
{table_context}
""").content
        if isinstance(response, list):
            raw_suggestions = response
        else:
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(response).strip(), flags=re.IGNORECASE)
            raw_suggestions = json.loads(cleaned)
        excluded = {re.sub(r"[^a-z0-9]+", " ", item.casefold()).strip() for item in previous_suggestions if isinstance(item, str)}
        suggestions = []
        for item in raw_suggestions:
            if not isinstance(item, str) or not item.strip():
                continue
            question = item.strip()
            normalized = re.sub(r"[^a-z0-9]+", " ", question.casefold()).strip()
            if normalized in excluded or any(
                normalized in re.sub(r"[^a-z0-9]+", " ", existing.casefold()).strip()
                or re.sub(r"[^a-z0-9]+", " ", existing.casefold()).strip() in normalized
                for existing in suggestions
            ):
                continue
            suggestions.append(question)
        # If the model repeats earlier ideas, construct fresh questions directly from
        # real metadata rather than showing duplicate chips again.
        for table in metadata:
            cols = table["columns"]
            numeric = next((c["name"] for c in cols if any(t in c["type"].lower() for t in ("int", "numeric", "decimal", "real", "double", "float"))), None)
            time_col = next((c["name"] for c in cols if any(t in c["type"].lower() for t in ("date", "time"))), None)
            category = next((c["name"] for c in cols if c["name"] != numeric and any(t in c["type"].lower() for t in ("character", "text", "boolean", "enum"))), None)
            candidates = [f"How many rows are in {table['table']}?"]
            if category:
                candidates.extend([
                    f"Which {category} values appear most often in {table['table']}?",
                    f"How many records in {table['table']} have each {category}?",
                ])
            if numeric:
                candidates.extend([
                    f"What are the highest and lowest {numeric} values in {table['table']}?",
                    f"What is the average {numeric} in {table['table']}?",
                ])
                if category:
                    candidates.append(f"Compare average {numeric} across {category} in {table['table']}.")
                if time_col:
                    candidates.append(f"Show how {numeric} changes over {time_col} in {table['table']} as a chart.")
            if time_col:
                candidates.append(f"Show the most recent records from {table['table']} ordered by {time_col}.")
            for question in candidates:
                normalized = re.sub(r"[^a-z0-9]+", " ", question.casefold()).strip()
                if normalized not in excluded and all(
                    normalized != re.sub(r"[^a-z0-9]+", " ", existing.casefold()).strip()
                    for existing in suggestions
                ):
                    suggestions.append(question)
                    if len(suggestions) == 3:
                        break
            if len(suggestions) == 3:
                break
        return {"status": "success", "suggestions": suggestions[:3]}
    except HTTPException:
        raise
    except Exception as exc:
        return {"status": "error", "suggestions": [], "message": f"Could not generate suggestions: {exc}"}


@app.post("/api/fix-query")
async def fix_failed_query(request: Request):
    """Regenerate a failed read-only SQL query using its database error and schema."""
    data = await request.json()
    question = (data.get("question") or "").strip()
    failed_sql = (data.get("sql") or "").strip()
    error = (data.get("error") or "").strip()
    selected_dataset = (data.get("dataset") or "database").strip()
    if not question or not failed_sql or not error:
        raise HTTPException(status_code=400, detail="The question, failed query, and execution error are required.")
    if not re.match(r"(?is)^\s*(select|with)\b", failed_sql):
        raise HTTPException(status_code=400, detail="Only failed read-only queries can be repaired.")
    try:
        schema_info = DatabaseUtil(get_db_config()).schema_details("public")
        from utils.llm_pick import pick_llm
        repaired = pick_llm("medium").invoke(f"""
Repair this PostgreSQL query after the database returned an execution error.
Return only one corrected, read-only SQL statement. Preserve the user's intended analysis and use only fields present in the schema. Do not include markdown or explanation.

Original user request: {question}
Selected data source: {selected_dataset}
Failed SQL:
{failed_sql}

Database error:
{error}

Database schema:
{schema_info}
""").content
        if not isinstance(repaired, str):
            repaired = str(repaired)
        repaired = re.sub(r"^```(?:sql)?\s*|\s*```$", "", repaired.strip(), flags=re.IGNORECASE).strip()
        if not re.match(r"(?is)^(select|with)\b", repaired) or ";" in repaired.rstrip(";"):
            raise ValueError("The query repair did not return a single read-only SQL statement.")
        sql_result = structured_query(repaired)
        if sql_result.get("error"):
            return {"status": "error", "message": sql_result["error"], "sql": repaired}
        return {"status": "success", "sql": repaired, "result": sql_result}
    except HTTPException:
        raise
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
