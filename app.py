from flask import Flask, render_template, request, session
from utils import preprocess_and_save
import pandas as pd
from groq import Groq

app = Flask(__name__)
app.secret_key = "dev-secret-change-this"  # needed for session to work


@app.route("/", methods=["GET", "POST"])
def index():
    message = ""
    df = None
    df_preview_html = ""
    result_html = ""
    code_generated = ""

    if request.method == "POST":
        file = request.files.get("file")
        query = request.form.get("query")
        groq_key = request.form.get("api_key")

        if not groq_key:
            message = "Please enter your Groq API key."

        elif file and file.filename:
            # A new file was uploaded -> process it and remember where
            # the cleaned version lives on disk so we don't need it
            # re-uploaded for the next question.
            df, cols, df_html, err, temp_path = preprocess_and_save(file)
            if err:
                message = err
            else:
                session["cleaned_csv_path"] = temp_path
                df_preview_html = df.head().to_html(classes="table-auto w-full")

        elif session.get("cleaned_csv_path"):
            # No new file this time -> reuse the last cleaned file
            try:
                df = pd.read_csv(session["cleaned_csv_path"])
                df_preview_html = df.head().to_html(classes="table-auto w-full")
            except Exception as e:
                message = f"Couldn't reload your previous file: {e}"

        else:
            message = "Please upload a file first."

        # Only try to answer a question if we actually have data loaded
        if df is not None and query:
            try:
                prompt = f"""
You are a Python data analyst. Given a pandas DataFrame named `df`, write Python code using pandas to answer this question:

Question: {query}

Only return the Python code (no explanation). Use 'result' as the final output variable.
"""
                client = Groq(api_key=groq_key)
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="openai/gpt-oss-20b",  # llama-3.3-70b-versatile was deprecated Aug 2026
                )

                code_generated = chat_completion.choices[0].message.content.strip("`python").strip("`")

                local_vars = {"df": df}
                exec(code_generated, {}, local_vars)

                result = local_vars.get("result", "No result generated.")
                if isinstance(result, pd.DataFrame):
                    result_html = result.to_html(classes="table-auto w-full")
                else:
                    result_html = str(result)

            except Exception as e:
                message = f"Error running Groq code: {e}"

    return render_template(
        "index.html",
        message=message,
        df_preview_html=df_preview_html,
        code_generated=code_generated,
        result_html=result_html,
    )


if __name__ == "__main__":
    app.run(debug=True)
