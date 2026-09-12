# AI Data Analyst

A small Flask web app that lets you upload a CSV/Excel file, ask a question about it in plain English, and get an answer computed by pandas — the actual pandas code is generated on the fly by an LLM (via [Groq](https://console.groq.com)) and run against your real data.

## How it works

1. Upload a CSV or Excel file
2. Ask a question referencing your actual column names (e.g. "what is the average Glucose for Outcome = 1?")
3. The app sends your question + a prompt template to a Groq-hosted LLM, asking it to write pandas code that computes the answer
4. That generated code is executed against your DataFrame with `exec()`
5. The result, the generated code, and a preview of your data are all shown in the UI

## Setup

```bash
python -m venv venv
source venv/bin/activate   # venv\Scripts\activate on Windows
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000`, paste in a free [Groq API key](https://console.groq.com), upload a file, and ask away.

## Project structure

```
ai-data-analyst/
├── app.py              # Flask routes + LLM call + exec logic
├── utils.py            # File upload cleaning/preprocessing
├── requirements.txt
└── templates/
    └── index.html      # UI
```
# Page Preview
<img src="Preview_of_fileuploading" width=800>
