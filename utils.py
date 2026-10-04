import warnings
import pandas as pd


def _read(path):
    if path.lower().endswith(".csv"):
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin-1")
    return pd.read_excel(path)


def _clean(df):
    """Strip column names, drop empty rows, convert numeric and date columns."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all")

    for col in df.columns:
        s = df[col]
        if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
            continue
        non_null = s.notna().sum()
        if non_null == 0:
            continue

        numeric = pd.to_numeric(s, errors="coerce")
        if numeric.notna().sum() / non_null > 0.9:
            df[col] = numeric
            continue

        if any(k in col.lower() for k in ("date", "time")):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(s, errors="coerce")
            if parsed.notna().sum() / non_null > 0.8:
                df[col] = parsed
    return df


def df_to_html(obj, max_rows=200):
    """Render a DataFrame / Series / scalar as an HTML snippet."""
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        return obj.head(max_rows).to_html(classes="data-table", border=0, index=not isinstance(obj.index, pd.RangeIndex))
    return f'<pre class="p-4 text-lg">{obj}</pre>'


def preprocess_and_save(file_path):
    """Read the uploaded file, clean it, save a cleaned CSV next to it.

    Returns: df, clean_path, preview_html, err
    """
    try:
        df = _clean(_read(file_path))
        if df.empty:
            return None, None, None, "The file has no data."
        clean_path = file_path.rsplit(".", 1)[0] + "_clean.csv"
        df.to_csv(clean_path, index=False)
        return df, clean_path, df_to_html(df.head(5)), None
    except Exception as e:
        return None, None, None, f"Could not read file: {e}"


def load_clean(clean_path):
    return _clean(pd.read_csv(clean_path))
