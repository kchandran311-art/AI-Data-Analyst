import pandas as pd
import tempfile


def preprocess_and_save(file):
    """
    Takes an uploaded file (CSV or Excel), cleans it up a bit,
    saves it to a temporary CSV on disk, and returns everything
    the rest of the app needs.

    Returns: (df, columns, preview_html, error_message)
    """
    try:
        filename = file.filename

        # 1. Load the file into a DataFrame based on its extension
        if filename.endswith(".csv"):
            df = pd.read_csv(file, encoding="utf-8", na_values=["NA", "N/A", "missing"])
        elif filename.endswith(".xlsx"):
            df = pd.read_excel(file, na_values=["NA", "N/A", "missing"])
        else:
            return None, None, None, "Unsupported file format. Please upload a .csv or .xlsx file."

        # 2. Try to convert object columns that are secretly dates
        for col in df.select_dtypes(include=["object"]):
            try:
                converted = pd.to_datetime(df[col], errors="raise", infer_datetime_format=True)
                df[col] = converted
            except (ValueError, TypeError):
                pass  # not a date column, leave it alone

        # 3. Try to convert object columns that are secretly numeric
        for col in df.select_dtypes(include=["object"]):
            converted = pd.to_numeric(df[col], errors="coerce")
            # Only keep the conversion if it didn't wipe out real data
            if converted.notna().sum() >= df[col].notna().sum() * 0.9:
                df[col] = converted

        # 4. Save the cleaned data to a temp CSV so it persists across requests if needed
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline="") as temp_file:
            df.to_csv(temp_file.name, index=False, quoting=1)  # QUOTE_ALL
            saved_path = temp_file.name

        # 5. Build a small preview (first 5 rows) as an HTML table for the UI
        preview_html = df.head().to_html(classes="table-auto w-full")

        return df, df.columns.tolist(), preview_html, None, saved_path

    except Exception as e:
        return None, None, None, str(e), None
