import math
import re

import numpy as np
try:
    from groq import Groq
except ImportError:
    Groq = None

from config import GROQ_API_KEY
from knowledge_graph import KNOWLEDGE_BASE

SYSTEM_PROMPT = """
You are a helpful, polite, and concise Farmer assitant bot named Kissan bot for indian farmers. Your job is to assist users strictly based on the provided information.

Instructions:
I Want you to "ALWAYS" answer in Hinglish and never Translate your response 

Stick "STRICTLY" to the topic of Kisan Bot and do not go off-topic.

Use standard newlines ('\n') for paragraph breaks.

Do not use any HTML tags like <br> or markdown formatting like * or ** unless quoting exact UI text (e.g., 'Click the "Submit" button.').

Use numbered lists only when describing step-by-step instructions derived from the relevant information.

Keep responses short and to the point. Avoid unnecessary pleasantries.

Always include links directly in the response if provided .


"""

_qa_pairs = KNOWLEDGE_BASE
_tfidf_mat = None
_vocab = None
_idf = None
_SIM_THRESHOLD = 0.12

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


def _tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def _expand_terms(tokens):
    if not tokens:
        return []
    out = list(tokens)
    for i in range(len(tokens) - 1):
        out.append(f"{tokens[i]}_{tokens[i + 1]}")
    return out


def _build_tfidf(corpus):
    """L2-normalized TF-IDF rows; no scikit-learn (avoids SciPy/NumPy ABI issues)."""
    n_docs = len(corpus)
    doc_terms = [_expand_terms(_tokenize(t)) for t in corpus]
    df = {}
    for terms in doc_terms:
        for t in set(terms):
            df[t] = df.get(t, 0) + 1
    if not df:
        return None, None, None
    vocab = {t: i for i, t in enumerate(sorted(df.keys()))}
    vdim = len(vocab)
    idf = np.zeros(vdim)
    for t, i in vocab.items():
        idf[i] = math.log((n_docs + 1.0) / (df[t] + 1.0)) + 1.0

    mat = np.zeros((n_docs, vdim))
    for d, terms in enumerate(doc_terms):
        tf = {}
        for t in terms:
            tf[t] = tf.get(t, 0) + 1
        denom = max(len(terms), 1)
        for t, c in tf.items():
            if t in vocab:
                mat[d, vocab[t]] = (c / denom) * idf[vocab[t]]
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms
    return mat, vocab, idf


def _query_vec(text, vocab, idf):
    terms = _expand_terms(_tokenize(text))
    if not terms or not vocab:
        return None
    tf = {}
    for t in terms:
        tf[t] = tf.get(t, 0) + 1
    denom = max(len(terms), 1)
    row = np.zeros(len(vocab))
    for t, c in tf.items():
        if t in vocab:
            row[vocab[t]] = (c / denom) * idf[vocab[t]]
    n = np.linalg.norm(row)
    if n == 0:
        return None
    return row / n


def _fit_retrieval():
    global _tfidf_mat, _vocab, _idf
    if _tfidf_mat is not None or not _qa_pairs:
        return
    corpus = [f"{p['question']} {p['answer']}" for p in _qa_pairs]
    _tfidf_mat, _vocab, _idf = _build_tfidf(corpus)


def get_best_match(user_message):
    """Return the best-matching answer text from the knowledge base, or a sentinel string."""
    _fit_retrieval()
    user_message = user_message.strip().lower()
    if not user_message or not _qa_pairs or _tfidf_mat is None or _vocab is None:
        return "No relevant information found."
    qv = _query_vec(user_message, _vocab, _idf)
    if qv is None:
        return "No relevant information found."
    sims = _tfidf_mat @ qv
    idx = int(np.argmax(sims))
    if sims[idx] < _SIM_THRESHOLD:
        return "No relevant information found."
    return _qa_pairs[idx]["answer"]


def generate_response(user_message):
    """Generate AI response using Groq API, with TF-IDF retrieval over the knowledge base."""
    best_match = get_best_match(user_message)

    if client is None:
        if best_match != "No relevant information found.":
            return best_match
        return (
            "Kissan bot ke liye GROQ_API_KEY .env file mein set karo. "
            "Bina key ke bhi tum specific farming questions poochh sakte ho  -main knowledge base se match dhundhunga."
        )

    prompt = SYSTEM_PROMPT + "\nRelevant Information: " + best_match
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_message},
            ],
            model="openai/gpt-oss-120b",
        )

        bot_response = response.choices[0].message.content
        formatted_response = bot_response.replace("**Step", "<br><br>**Step")

        return formatted_response

    except Exception as e:
        print("❌ Error calling Groq API:", str(e))

        if "quota" in str(e).lower() or "rate limit" in str(e).lower():
            return (
                "We're currently experiencing a high volume of requests and have reached our daily response limit.\n"
                "Please try again later or contact FretBox support for urgent matters."
            )

        if best_match != "No relevant information found.":
            return best_match

        return "Apologies, we're currently unavailable to process your request. Please try again shortly."
