"""File based extraction and analysis helpers for the ETL agent."""
from __future__ import annotations

import json
import ast
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests


class ETLTools:
    ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = ROOT / "data"
    ALLOWED_FOLDERS = {"extract", "transform", "uploads"}
    FORMATS = {"csv", "json", "parquet"}

    def _resolve_file(self, file_path: str) -> Path:
        path = Path(file_path)
        if not path.is_absolute():
            path = self.ROOT / path
        path = path.resolve()
        if self.DATA_DIR.resolve() not in path.parents or path.suffix.lower().lstrip(".") not in self.FORMATS:
            raise ValueError("File must be a CSV, JSON, or Parquet file under data/.")
        if not path.is_file():
            raise FileNotFoundError(f"Data file not found: {path}")
        return path

    @staticmethod
    def _to_frame(payload):
        # Support common API response shapes: lists, {results: []}, {data: []},
        # and ordinary objects. Keep nested values in normalized columns.
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict):
            records = next((payload[k] for k in ("results", "data", "items", "records")
                            if isinstance(payload.get(k), list)), payload)
        else:
            records = {"value": payload}
        if isinstance(records, dict):
            return pd.json_normalize(records)
        if isinstance(records, list):
            return pd.json_normalize(records) if records else pd.DataFrame()
        return pd.json_normalize({"value": records})

    def extract_load(self, url: str, output_folder: str = "data/extract", format: str = "csv") -> str:
        fmt = format.lower().lstrip(".")
        if fmt not in self.FORMATS:
            return f"Unsupported format: {format}. Choose csv, json, or parquet."
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return "Failed to extract data: a valid http(s) URL is required."
        folder = Path(output_folder)
        if not folder.is_absolute():
            folder = self.ROOT / folder
        folder = folder.resolve()
        if self.DATA_DIR.resolve() not in folder.parents or folder.relative_to(self.DATA_DIR.resolve()).parts[0] not in self.ALLOWED_FOLDERS:
            return "Failed to extract data: output folder must be under data/extract, data/transform, or data/uploads."
        try:
            pages = []
            next_url = url
            visited = set()
            for _ in range(50):
                if next_url in visited:
                    break
                visited.add(next_url)
                response = requests.get(next_url, timeout=(5, 30), headers={"Accept": "application/json"})
                response.raise_for_status()
                payload = response.json()
                pages.append(self._to_frame(payload))
                following = payload.get("next") if isinstance(payload, dict) else None
                if not following:
                    break
                next_url = urljoin(response.url, str(following))
                if urlparse(next_url).scheme not in {"http", "https"}:
                    raise ValueError("API returned an invalid pagination URL")
            frame = pd.concat(pages, ignore_index=True) if pages else pd.DataFrame()
            folder.mkdir(parents=True, exist_ok=True)
            stem = re.sub(r"[^A-Za-z0-9_-]+", "_", parsed.path.rstrip("/").split("/")[-1] or "api_data")[:80]
            target = folder / f"{stem}.{fmt}"
            if fmt == "csv":
                frame.to_csv(target, index=False)
            elif fmt == "json":
                frame.to_json(target, orient="records", lines=True)
            else:
                frame.to_parquet(target, index=False)
            relative_target = target.relative_to(self.ROOT).as_posix()
            return f"Data successfully extracted and saved to {relative_target} ({len(frame)} rows, {len(frame.columns)} columns)."
        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
            return f"Failed to extract data: {exc}"
        except Exception as exc:
            return f"Failed to save extracted data: {exc}"

    def read_frame(self, file_path: str) -> pd.DataFrame:
        path = self._resolve_file(file_path)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix == ".json":
            try:
                return pd.read_json(path, lines=True)
            except ValueError:
                return pd.read_json(path)
        return pd.read_parquet(path)

    def transform_load_context(self, file_path: str) -> str:
        frame = self.read_frame(file_path)
        return f"Columns: {list(frame.columns)}\nRows: {len(frame)}\nSample:\n{frame.head(5).to_string(index=False)}"

    def execute_code(self, code: str, frame: pd.DataFrame, output_path: str | None = None) -> str:
        """Run generated analysis with a preloaded dataframe and bounded output location."""
        output = Path(output_path or (self.DATA_DIR / "extract" / "analysis-output.csv")).resolve()
        if self.DATA_DIR.resolve() not in output.parents:
            return "Failed to execute code: output must remain under data/."
        try:
            tree = ast.parse(code, mode="exec")
            forbidden = {"__", "to_pickle", "to_sql", "to_csv", "to_json", "to_parquet", "to_excel", "to_feather", "to_hdf"}
            safe_dataframe_methods = {"head", "tail", "describe", "info", "groupby", "agg", "aggregate", "mean", "sum", "count", "min", "max", "median", "nunique", "unique", "value_counts", "sort_values", "sort_index", "nlargest", "nsmallest", "isin", "isna", "notna", "fillna", "dropna", "astype", "round", "loc", "iloc", "reset_index", "rename", "select_dtypes", "to_string", "to_dict"}
            if output_path is not None:
                safe_dataframe_methods |= {"to_csv", "to_json", "to_parquet"}
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom, ast.Lambda, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    raise ValueError("imports and definitions are not allowed")
                if isinstance(node, ast.Name) and node.id.startswith("_"):
                    raise ValueError("private names are not allowed")
                if isinstance(node, ast.Attribute) and (node.attr.startswith("_") or (output_path is None and node.attr in forbidden)):
                    raise ValueError("private or file-writing attributes are not allowed")
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id not in {"len", "min", "max", "sum", "sorted", "list", "dict", "str", "int", "float", "round", "enumerate", "print"}:
                        raise ValueError("function call is not allowed")
                    if isinstance(node.func, ast.Attribute) and node.func.attr not in safe_dataframe_methods | {"to_datetime"}:
                        raise ValueError("dataframe method is not allowed")
                    if not isinstance(node.func, (ast.Name, ast.Attribute)):
                        raise ValueError("dynamic calls are not allowed")
        except Exception as exc:
            return f"Failed to execute code: {exc}"
        safe_builtins = {"print": print, "len": len, "range": range, "min": min, "max": max,
                         "sum": sum, "sorted": sorted, "list": list, "dict": dict, "str": str,
                         "int": int, "float": float, "round": round, "enumerate": enumerate}
        namespace = {"__builtins__": safe_builtins, "pd": pd, "df": frame.copy(), "output_path": str(output)}
        try:
            exec(compile(code, "<etl-analysis>", "exec"), namespace, namespace)
            result = namespace.get("result")
            if isinstance(result, pd.DataFrame):
                return result.head(100).to_string(index=False)
            if result is not None:
                return str(result)[:10000]
            return "Code executed successfully."
        except Exception as exc:
            return f"Failed to execute code: {exc}"
