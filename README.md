# Data AI Agent

A highly capable AI-powered assistant designed to query PostgreSQL databases, extract data from external APIs, and analyze local datasets seamlessly in your browser. Powered by LangGraph, Anthropic Claude, and FastAPI.

## Features

- **Conversational Analytics**: Chat with your data using plain English.
- **SQL Analysis**: Automatically generate, execute, and interpret PostgreSQL queries.
- **ETL Workflows**: Extract data from public APIs or upload your own files (CSV, JSON, Parquet) for instant Pandas-based analysis.
- **Intelligent Routing**: Queries are automatically routed to either the SQL agent (for your main database) or the ETL agent (for uploaded files and API endpoints).
- **Session Ledger**: Keep track of your past questions and their outcomes.
- **Data Store UI**: View a unified ledger of files extracted, transformed, and uploaded.

## Architecture

- **Backend**: FastAPI
- **Agent Orchestration**: LangGraph (`StateGraph`)
- **LLM**: Anthropic Claude 3 (`langchain_anthropic`)
- **Database Utilities**: Psycopg2 for PostgreSQL queries
- **Data Engineering**: Pandas for runtime transformations

## Prerequisites

- Python 3.9+
- PostgreSQL Server
- An Anthropic API Key

## Quick Start

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <repository-directory>
```

### 2. Set up virtual environment and dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Environment Variables

Create a `.env` file in the root directory and configure the following credentials:

```env
POSTGRES_USER=your_postgres_user
POSTGRES_PASSWORD=your_postgres_password
POSTGRES_DB=your_database_name
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
ANTHROPIC_API_KEY=your_anthropic_api_key
```

For Vercel + Supabase, prefer a Supabase pooled Postgres connection string:

```env
DATABASE_URL=postgresql://postgres.<project-ref>:<password>@aws-0-<region>.pooler.supabase.com:6543/postgres?sslmode=require
OPENROUTER_API_KEY=your_openrouter_api_key
```

The app also still supports the local `host`, `port`, `database`, `user`, and `password` variables.

### 4. Run the Application

Start the FastAPI application via the `main.py` entry point:

```bash
python main.py
```
*(Alternatively, you can run `uvicorn api:app --reload`)*

The application will start on `http://localhost:8000`.

## Deploy to Vercel with Supabase

1. Create a Supabase project and copy the pooled Postgres connection string from Project Settings -> Database.
2. Seed Supabase from this project:

```bash
DATABASE_URL="<your-supabase-postgres-url>" python feed_db.py
```

3. Log in and deploy with Vercel:

```bash
npx vercel login
npx vercel env add DATABASE_URL production
npx vercel env add OPENROUTER_API_KEY production
npx vercel deploy --prod
```

`.vercelignore` excludes local datasets and `.env` files from the upload. Runtime file uploads/history on Vercel are ephemeral; keep durable data in Supabase or another external store.

### 5. Using the Application

1. **Ask a Question**: Open `http://localhost:8000` in your browser. Use the main search bar to ask a plain English question about your database.
2. **Attach Data**: Click the paperclip icon in the search bar to upload local CSV, JSON, or Parquet files. They will be seamlessly loaded into the `data/uploads/` directory for ETL analysis.
3. **External APIs**: Ask the agent to "Extract data from `<api-url>` and save as CSV" to ingest public data.
4. **My Files Ledger**: Click on the "My Files" tab to view data that has been extracted, transformed, or uploaded manually.
