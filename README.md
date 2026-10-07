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

### 4. Run the Application

Start the FastAPI application via the `main.py` entry point:

```bash
python main.py
```
*(Alternatively, you can run `uvicorn api:app --reload`)*

The application will start on `http://localhost:8000`.

### 5. Using the Application

1. **Ask a Question**: Open `http://localhost:8000` in your browser. Use the main search bar to ask a plain English question about your database.
2. **Attach Data**: Click the paperclip icon in the search bar to upload local CSV, JSON, or Parquet files. They will be seamlessly loaded into the `data/uploads/` directory for ETL analysis.
3. **External APIs**: Ask the agent to "Extract data from `<api-url>` and save as CSV" to ingest public data.
4. **My Files Ledger**: Click on the "My Files" tab to view data that has been extracted, transformed, or uploaded manually.
