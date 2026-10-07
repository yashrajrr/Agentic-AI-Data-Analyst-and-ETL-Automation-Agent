
import os, sys
from pathlib import Path
cwd = Path(sys.argv[1]); os.chdir(cwd)
from dotenv import dotenv_values
for k,v in dotenv_values(Path(cwd)/".env").items():
    os.environ[k] = v or ""
# Explicitly override the known-broken os.environ port with dotenv truth:
os.environ["port"] = os.environ.get("port","5432")
print("NEW PROCESS db port:", os.environ.get("port","MISSING"), flush=True)
print("NEW PROCESS db host:", os.environ.get("host","MISSING"), flush=True)
print("NEW PROCESS db user:", os.environ.get("user","MISSING"), flush=True)
import uvicorn
uvicorn.run("api:app", host="127.0.0.1", port=8000, log_level="info")
