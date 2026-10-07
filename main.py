"""
main.py – Ola Data AI Agent Launcher
======================================
Choose how you want to run the project:

  [1] API Server      – Start the FastAPI backend + Web UI (http://127.0.0.1:8000)
  [2] Agent CLI       – Chat with the Data Agent directly in the terminal
  [3] Feed Database   – Load CSV data into PostgreSQL (resets existing data)
  [4] Exit
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env from the project root ──────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


# ── Helpers ───────────────────────────────────────────────────────────────────

BANNER = r"""
╔══════════════════════════════════════════════════════╗
║          🚕  Ola Data AI Agent  Launcher             ║
╠══════════════════════════════════════════════════════╣
║  [1]  API Server   – FastAPI + Web UI                ║
║  [2]  Agent CLI    – Chat in the terminal            ║
║  [3]  Feed DB      – Load CSVs into PostgreSQL       ║
║  [4]  Exit                                           ║
╚══════════════════════════════════════════════════════╝
"""

def prompt_choice() -> str:
    print(BANNER)
    choice = input("Enter your choice (1/2/3/4): ").strip()
    return choice


# ── Mode 1 – API Server ───────────────────────────────────────────────────────

def run_api_server():
    """Launch uvicorn serving api:app."""
    host = "127.0.0.1"
    port = "8000"
    print(f"\n🚀  Starting API server → http://{host}:{port}")
    print("    (Press Ctrl+C to stop)\n")
    try:
        subprocess.run(
            [sys.executable, "-m", "uvicorn", "api:app",
             "--host", host, "--port", port, "--reload"],
            cwd=str(ROOT),
            check=True,
        )
    except KeyboardInterrupt:
        print("\n⏹  Server stopped.")
    except subprocess.CalledProcessError as exc:
        print(f"\n❌  Server exited with code {exc.returncode}")


# ── Mode 2 – Agent CLI ────────────────────────────────────────────────────────

def run_agent_cli():
    """Interactive terminal chat with the Data Agent."""
    print("\n🤖  Agent CLI ready. Type 'exit' or 'quit' to stop.\n")

    # Import lazily so missing deps surface with a clear message.
    try:
        from agents.data_agent import data_agent
        from langchain_core.messages import HumanMessage
    except ImportError as exc:
        print(f"❌  Import error: {exc}")
        print("    Make sure your virtual environment is active and dependencies are installed.")
        return

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋  Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("👋  Goodbye!")
            break

        print("\n⏳  Thinking …\n")
        try:
            response = data_agent.invoke(
                {
                    "messages": [HumanMessage(content=user_input)],
                    "route_response": "",
                }
            )

            route = response.get("route_response", "unknown")
            messages = response.get("messages", [])

            # Extract a readable answer from the last message
            last = messages[-1] if messages else None
            answer = ""

            if route == "sql" and isinstance(last, dict):
                answer = last.get("final_answer") or last.get("sql_query_execution_result") or ""
            elif route == "etl" and isinstance(last, dict):
                inner = last.get("messages", [])
                for msg in reversed(inner):
                    content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
                    if content and isinstance(content, str) and content.strip():
                        answer = content
                        break

            if not answer:
                # Fallback: last message with non-empty string content
                for msg in reversed(messages):
                    content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
                    if content and isinstance(content, str) and content.strip():
                        answer = content
                        break

            print(f"[Route: {route.upper()}]\n")
            print(f"Agent: {answer or '(no text response)'}\n")
            print("─" * 60)

        except Exception as exc:
            print(f"❌  Error: {exc}\n")


# ── Mode 3 – Feed Database ────────────────────────────────────────────────────

def run_feed_db():
    """Run feed_db.py to (re-)load all CSVs into PostgreSQL."""
    print("\n📦  Loading CSVs into PostgreSQL …\n")
    confirm = input("⚠️  This will TRUNCATE existing data. Continue? (yes/no): ").strip().lower()
    if confirm not in {"yes", "y"}:
        print("    Cancelled.")
        return

    try:
        subprocess.run(
            [sys.executable, "feed_db.py"],
            cwd=str(ROOT),
            check=True,
        )
        print("\n✅  Database seeded successfully!")
    except subprocess.CalledProcessError as exc:
        print(f"\n❌  feed_db.py exited with code {exc.returncode}")


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    dispatch = {
        "1": run_api_server,
        "2": run_agent_cli,
        "3": run_feed_db,
        "4": lambda: print("👋  Goodbye!"),
    }

    choice = prompt_choice()
    action = dispatch.get(choice)

    if action is None:
        print(f"\n❌  Invalid choice '{choice}'. Please run again and enter 1–4.")
        sys.exit(1)

    action()


if __name__ == "__main__":
    main()