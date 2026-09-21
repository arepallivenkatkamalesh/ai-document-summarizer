import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError
import pypdf
import logging
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("google_genai").setLevel(logging.ERROR)
logging.getLogger("google").setLevel(logging.ERROR)
# Load API key from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: GEMINI_API_KEY not found. Check your .env file.")
    sys.exit(1)

client = genai.Client(api_key=api_key)


def get_active_model() -> str:
    """Finds the first available text-generation model for this key."""
    try:
        for m in client.models.list():
            # Match active models supporting content generation
            name = getattr(m, "name", "")
            actions = getattr(m, "supported_actions", []) or []
            if "generateContent" in actions or "generate_content" in actions:
                return name
        return "gemini-1.5-flash"
    except Exception:
        return "gemini-1.5-flash"


ACTIVE_MODEL = get_active_model()
print(f"[*] Using active model: {ACTIVE_MODEL}")


def extract_text_from_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' does not exist.")

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    elif ext == ".pdf":
        reader = pypdf.PdfReader(file_path)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Use .txt or .pdf.")

    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Document is empty or contains no readable text.")
    return cleaned


import time
import re

def clean_markdown(text: str) -> str:
    # Remove bold/italic asterisks
    text = re.sub(r"\*{1,3}", "", text)
    # Remove markdown headers like ###
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    return text.strip()

def query_llm(prompt: str) -> str:
    models_to_try = ["gemini-3.6-flash", "gemini-2.5-pro"]
    for model_name in models_to_try:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                return clean_markdown(response.text)
            except APIError as e:
                if "high demand" in str(e).lower() or e.code == 503:
                    time.sleep(2)
                    continue
                return f"[API Error]: {e.message}"
            except Exception as e:
                return f"[System Error]: {str(e)}"
    return "[Error]: Server busy across all models. Please retry in a few moments."
    try:
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
        )
        return response.text
    except APIError as e:
        return f"[API Error]: {e.message}"
    except Exception as e:
        return f"[System Error]: {str(e)}"


def summarize_document(doc_text: str) -> str:
    prompt = f"Provide a clear, structured summary with bullet points:\n\n{doc_text}"
    return query_llm(prompt)


def answer_query(doc_text: str, question: str) -> str:
    prompt = f"Answer using ONLY this document context:\n{doc_text}\n\nQuestion: {question}"
    return query_llm(prompt)


def main():
    print("=" * 60)
    print(" AI Text Summarization & Q&A Assistant (CLI)")
    print("=" * 60)

    while True:
        file_path = input("\nEnter document path (.txt or .pdf): ").strip().strip('"').strip("'")
        if file_path.lower() in ["exit", "quit"]:
            return
        try:
            document_text = extract_text_from_file(file_path)
            print(f" Loaded document ({len(document_text)} characters).")
            break
        except Exception as err:
            print(f" [Error]: {err}")

    print("\n--- Generating Document Summary ---")
    print(summarize_document(document_text))

    print("\n" + "=" * 60)
    print(" Interactive Q&A Mode (Type 'exit' to quit)")
    print("=" * 60)

    while True:
        q = input("\nAsk a question about this document: ").strip()
        if not q:
            continue
        if q.lower() in ["exit", "quit"]:
            break
        print("\nThinking...")
        print(f"\nAssistant:\n{answer_query(document_text, q)}")


if __name__ == "__main__":
    main()