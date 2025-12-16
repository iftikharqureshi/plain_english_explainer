# Plain-English Explainer for Dense Paragraphs

Streamlit app that rewrites dense academic or technical paragraphs into concise, schema-validated JSON output using an OpenAI fine-tuned model.

## Features
- Streamlit UI for pasting paragraph text and triggering an explanation.
- Calls a fine-tuned OpenAI chat model and validates the JSON response against an embedded JSON Schema.
- Structured output: 3 summary sentences, 5 bullet points, 3 vocabulary terms, and optional evidence lines.

## Prerequisites
- Python 3.10+ recommended.
- An OpenAI API key with access to the fine-tuned model.

## Setup
1) Clone the repository and create a virtual environment (optional but recommended):
```bash
python -m venv .venv
source .venv/bin/activate
```
2) Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
3) Configure environment variables:
```bash
cp .env.example .env
# edit .env and add your OpenAI API key
```
Required variable:
- `OPENAI_API_KEY`

## Running the app
```bash
streamlit run app.py
```
The app opens in your browser. Paste a paragraph, click **Explain paragraph**, and review the structured summary.

## Model configuration
The app references a fine-tuned model name in `app.py` (`MODEL`). Replace this with your own model identifier if needed.

## Project structure
- `app.py` — Streamlit application and OpenAI integration.
- `requirements.txt` — Python dependencies.
- `.env.example` — Template for environment variables.

## License
MIT License. See `LICENSE` for details.
