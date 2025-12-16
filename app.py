# Plain-English Explainer for Dense Paragraphs

import json
import os
import re
from typing import Dict, List

import openai
import streamlit as st
from jsonschema import Draft202012Validator, ValidationError
from openai import OpenAI

# ------------------------------ App config ----------------------------------

# Configure Streamlit page metadata for the explainer app
st.set_page_config(
    page_title="Plain-English Explainer for Dense Paragraphs", layout="centered"
)

# Using fine tuned model
MODEL = "ft:gpt-4.1-mini-2025-04-14:personal:plain-explainer-json-v1:CWoHgyO5"


# ------------------------------ Embedded schema ------------------------------
# JSON Schema that the model's response must satisfy
SCHEMA_TEXT = r"""
{
  "title": "ExplainerOutput",
  "type": "object",
  "required": ["summary_sentences", "bullets", "vocab"],
  "properties": {
    "summary_sentences": {
      "type": "array",
      "minItems": 3,
      "maxItems": 3,
      "items": { "type": "string" }
    },
    "bullets": {
      "type": "array",
      "minItems": 5,
      "maxItems": 5,
      "items": { "type": "string" }
    },
    "vocab": {
      "type": "array",
      "minItems": 3,
      "maxItems": 3,
      "items": {
        "type": "object",
        "required": ["term", "definition"],
        "properties": {
          "term": { "type": "string" },
          "definition": { "type": "string" }
        }
      }
    },
    "evidence_lines": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["bullet_index", "evidence"],
        "properties": {
          "bullet_index": { "type": "integer", "minimum": 0, "maximum": 4 },
          "evidence": { "type": "string" }
        }
      }
    }
  }
}
""".strip()

try:
    SCHEMA: Dict = json.loads(SCHEMA_TEXT)
except json.JSONDecodeError as e:
    raise RuntimeError(
        f"Embedded SCHEMA_TEXT is invalid JSON: {e.msg} (line {e.lineno}, col {e.colno})"
    ) from e


# ------------------------------ OpenAI client --------------------------------


@st.cache_resource(show_spinner=False)
def get_client() -> OpenAI:
    """Create and cache a configured OpenAI client."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
    return OpenAI(api_key=api_key)


# System prompt constrains the model to emit JSON only
SYSTEM_INSTRUCTION = (
    "You are a careful rewriting model. "
    "Output ONLY a single JSON object that follows the provided JSON Schema. "
    "Do not include any text before or after the JSON."
)


# ------------------------------ Core function --------------------------------


def explain_paragraph(paragraph_text: str) -> Dict:
    """
    Ask OpenAI for a JSON-only response and validate it locally against SCHEMA.
    Raises ValidationError/JSONDecodeError/RuntimeError on failure.
    """
    client = get_client()

    # Build a verbose user prompt that includes requirements and schema
    user_message = f"""
TASK
Rewrite the following dense paragraph into plain English without adding outside facts or opinions.

OUTPUT
Return ONE JSON object with:
- summary_sentences: exactly 3 sentences in plain English.
- bullets: exactly 5 short points, each drawn directly from the paragraph.
- vocab: exactly 3 items, each with "term" and "definition" taken from the paragraph.
- evidence_lines: OPTIONAL array of {{ bullet_index, evidence }} pairs (only include if helpful).

RULES
- Neutral tone. No advice. No opinions.
- Keep sentences short and clear.
- Do not output anything outside the JSON object.
- Follow the JSON Schema exactly.

JSON SCHEMA
{SCHEMA_TEXT}

PARAGRAPH
{paragraph_text.strip()}
""".strip()

    # Send the system and user messages to the fine-tuned chat model
    chat = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_message},
        ],
    )

    content: str = chat.choices[0].message.content or ""

    # Safety: Strip accidental code fences if they slip through.
    if content.startswith("```"):
        content = re.sub(
            r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE | re.DOTALL
        )

    # Parse JSON response and verify it matches the embedded schema
    obj: Dict = json.loads(content)
    Draft202012Validator(SCHEMA).validate(obj)
    return obj


# ------------------------------ UI -------------------------------------------

# Render page header and helper text for the explainer
st.title("📝 Plain-English Explainer for Dense Paragraphs")
st.write(
    (
        "Paste a dense paragraph or article excerpt and get a concise,"
        " structured summary with key terms."
    )
)

# Accept paragraph input from the user
paragraph: str = st.text_area(
    "Paragraph",
    height=200,
    placeholder="Paste your dense paragraph here…",
)

# Trigger explanation only after the button is pressed
if st.button("Explain paragraph", type="primary"):
    if not paragraph.strip():
        st.warning("Please paste a paragraph first.")
    else:
        with st.spinner("Calling OpenAI…"):
            try:
                result = explain_paragraph(paragraph)

            # JSON/Schema issues (local)
            except (ValidationError, json.JSONDecodeError) as e:
                st.error("Couldn't produce a valid JSON result.")
                with st.expander("Error details"):
                    st.exception(e)

            # OpenAI API / network issues
            except (
                openai.AuthenticationError,
                openai.PermissionDeniedError,
                openai.BadRequestError,
                openai.NotFoundError,
                openai.UnprocessableEntityError,
                openai.RateLimitError,
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.InternalServerError,
                openai.APIStatusError,
            ) as e:
                st.error("OpenAI API request failed.")
                with st.expander("Error details"):
                    st.exception(e)

            # Raised errors (e.g., missing OPENAI_API_KEY)
            except RuntimeError as e:
                st.error(str(e))
                with st.expander("Error details"):
                    st.exception(e)

            else:
                st.success("Done!")

                # Render the structured result (if keys exist).
                summary: List[str] = result.get("summary_sentences", [])
                bullets: List[str] = result.get("bullets", [])
                vocab: List[Dict[str, str]] = result.get("vocab", [])
                evidence_lines: List[Dict[str, str]] = result.get("evidence_lines", [])

                if summary:
                    st.subheader("Summary (3 sentences)")
                    for s in summary:
                        st.write(f"- {s}")

                if bullets:
                    st.subheader("Key points (5 bullets)")
                    st.write("\n".join(f"- {b}" for b in bullets))

                if vocab:
                    st.subheader("Vocabulary (3 terms)")
                    for item in vocab:
                        st.write(
                            f"**{item.get('term','')}** — {item.get('definition','')}"
                        )

                if evidence_lines:
                    st.subheader("Evidence lines (optional)")
                    for i, row in enumerate(evidence_lines):
                        st.markdown(f"{i+1}. {row.get('evidence')}")
