
import os, sys, time, socket, subprocess
from pathlib import Path
cwd = Path('D:\\Level Up\\ola data ai agent'); os.chdir(cwd)
from dotenv import dotenv_values
for k,v in dotenv_values(Path(cwd)/".env").items():
    os.environ.setdefault(k, v or "")
print("LAUNCHER db port:", os.environ.get("port","MISSING"), flush=True)
print("LAUNCHER db host:", os.environ.get("host","MISSING"), flush=True)
proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", "8000"],
    cwd=str(cwd),
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)
print("uvicorn pid", proc.pid, flush=True)
try:
    proc.wait(timeout=120)
    print("uvicorn exited", proc.returncode, flush=True)
    print(proc.stdout.read(), flush=True)
except subprocess.TimeoutExpired:
    print("uvicorn still running after 120s", flush=True)
    proc.kill()
    print(proc.stdout.read(), flush=True)
