import ast
import builtins
import os
import re
import tempfile
import uuid

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, render_template, request, session
from groq import Groq

from utils import df_to_html, load_clean, preprocess_and_save

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-change-me")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB uploads

MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "ai_data_analyst")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------- safety checks for LLM generated code ----------
ALLOWED_IMPORTS = {"pandas", "numpy"}
BLOCKED_NAMES = {
    "open", "exec", "eval", "compile", "input", "__import__", "globals",
    "locals", "vars", "getattr", "setattr", "delattr", "os", "sys",
    "subprocess", "shutil", "socket", "requests",
}
BLOCKED_ATTRS = {
    "read_csv", "read_excel", "read_json", "read_pickle", "read_sql", "read_html",
    "to_csv", "to_excel", "to_pickle", "to_sql", "to_parquet", "to_json", "eval",
}


def check_code(code):
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    raise ValueError(f"Import of '{a.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                raise ValueError(f"Import from '{node.module}' is not allowed")
        elif isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            raise ValueError(f"Use of '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in BLOCKED_ATTRS:
                raise ValueError(f"Use of '.{node.attr}' is not allowed")


def _safe_import(name, *args, **kwargs):
    if name.split(".")[0] not in ALLOWED_IMPORTS:
        raise ImportError(f"Import of '{name}' is not allowed")
    return builtins.__import__(name, *args, **kwargs)


def run_code(code, df):
    check_code(code)
    safe_builtins = {k: getattr(builtins, k) for k in (
        "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int",
        "isinstance", "len", "list", "map", "max", "min", "range", "reversed",
        "round", "set", "sorted", "str", "sum", "tuple", "zip", "print",
        "True", "False", "None", "Exception", "ValueError", "KeyError",
    )}
    safe_builtins["__import__"] = _safe_import
    env = {"__builtins__": safe_builtins, "pd": pd, "np": np, "df": df.copy()}
    exec(code, env)
    if "result" not in env:
        raise ValueError("The code did not create a variable named 'result'")
    return env["result"]


# ---------- Groq ----------
def build_prompt(df, query, bad_code=None, error=None):
    schema = "\n".join(f"- {c} ({t})" for c, t in df.dtypes.astype(str).items())
    sample = df.head(5).to_string()
    prompt = f"""You are a Python data analyst. A pandas DataFrame named df is already loaded.

Columns and dtypes:
{schema}

First 5 rows:
{sample}

Rules:
- Use only the exact column names above, with bracket syntax like df['Order ID'].
- Call methods with parentheses, e.g. df.groupby('Category')['Amount'].sum().
- Do not read or write files. Only import pandas as pd or numpy as np if needed.
- Store the final answer in a variable named result (a DataFrame, Series or single value).
- Return only Python code, no explanation.

Question: {query}"""
    if bad_code and error:
        prompt += f"\n\nYour previous code failed.\nCode:\n{bad_code}\nError: {error}\nReturn corrected code."
    return prompt


def extract_code(raw):
    m = re.search(r"```(?:python)?\s*(.*?)```", raw, re.DOTALL)
    return (m.group(1) if m else raw).strip()


def ask_groq(client, df, query, bad_code=None, error=None):
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[{"role": "user", "content": build_prompt(df, query, bad_code, error)}],
    )
    return extract_code(resp.choices[0].message.content)


def analyze(client, df, query):
    """Returns code, result, error. Retries once if the first code fails."""
    code, error = None, None
    for _ in range(2):
        code = ask_groq(client, df, query, bad_code=code if error else None, error=error)
        try:
            return code, run_code(code, df), None
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
    return code, None, f"Error running Groq code: {error}"


# ---------- routes ----------
@app.route("/", methods=["GET", "POST"])
def index():
    ctx = {"error": None, "info": None, "query": "", "preview_html": None,
           "code_generated": None, "result_html": None}
    df = None

    clean_path = session.get("clean_path")
    if clean_path and os.path.exists(clean_path):
        df = load_clean(clean_path)
    elif clean_path:
        session.pop("clean_path", None)
        ctx["info"] = "Your previous dataset expired. Please upload it again."

    if request.method == "POST":
        ctx["query"] = request.form.get("query", "").strip()
        file = request.files.get("file")

        if file and file.filename:
            ext = os.path.splitext(file.filename)[1].lower()
            if ext not in (".csv", ".xlsx"):
                ctx["error"] = "Please upload a .csv or .xlsx file."
            else:
                raw_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
                file.save(raw_path)
                new_df, new_clean, _, err = preprocess_and_save(raw_path)
                os.remove(raw_path)
                if err:
                    ctx["error"] = err
                else:
                    df = new_df
                    session["clean_path"] = new_clean
                    ctx["info"] = f"File processed: {len(df)} rows, {len(df.columns)} columns."

        if not ctx["error"] and ctx["query"]:
            api_key = os.environ.get("GROQ_API_KEY")
            if df is None:
                ctx["error"] = "Upload a dataset first."
            elif not api_key:
                ctx["error"] = "Server is missing GROQ_API_KEY."
            else:
                code, result, err = analyze(Groq(api_key=api_key), df, ctx["query"])
                ctx["code_generated"] = code  # shown even when there is an error
                if err:
                    ctx["error"] = err
                else:
                    ctx["result_html"] = df_to_html(result)

    if df is not None:
        ctx["preview_html"] = df_to_html(df.head(5))
    return render_template("index.html", **ctx)


if __name__ == "__main__":
    app.run(debug=True)
