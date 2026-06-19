import os
import sys
import pickle
import argparse
import textwrap
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── CONFIG ───────────────────────────────────────────────────────────────────
CHUNK_SIZE             = 200    # words per chunk
CHUNK_OVERLAP          = 40     # word overlap between consecutive chunks
TOP_K                  = 4      # chunks retrieved per query
INDEX_PATH             = "tfidf_index.pkl"
UNANSWERABLE_THRESHOLD = 0.10   # cosine score floor; below → flag unanswerable
GEMINI_MODEL           = "gemini-1.5-flash"


# ─── 1. CHUNKING ──────────────────────────────────────────────────────────────
def chunk_document(text: str, doc_id: str) -> list[dict]:
    """
    Sliding-window word-level chunking.
    200-word windows with 40-word overlap.
    Drops trailing fragments < 20 words.
    """
    words = text.split()
    step  = CHUNK_SIZE - CHUNK_OVERLAP
    chunks = []
    for i in range(0, len(words), step):
        window = words[i : i + CHUNK_SIZE]
        if len(window) < 20:
            break
        chunks.append({
            "doc_id":     doc_id,
            "chunk_id":   f"{doc_id}_c{i}",
            "start_word": i,
            "text":       " ".join(window),
        })
    return chunks


def load_corpus(corpus_dir: str) -> list[dict]:
    all_chunks = []
    for path in sorted(Path(corpus_dir).glob("*.txt")):
        text   = path.read_text(encoding="utf-8")
        doc_id = path.stem
        chunks = chunk_document(text, doc_id)
        all_chunks.extend(chunks)
        print(f"  [{doc_id}]  →  {len(chunks)} chunk(s)")
    if not all_chunks:
        raise FileNotFoundError(f"No .txt files found in {corpus_dir}")
    return all_chunks


# ─── 2. INDEXING (TF-IDF) ─────────────────────────────────────────────────────
def build_index(chunks: list[dict]) -> dict:
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),   # unigrams + bigrams for phrase matching
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,    # log(1+tf) dampens high-frequency terms
    )
    matrix = vectorizer.fit_transform(texts)
    return {"vectorizer": vectorizer, "matrix": matrix, "chunks": chunks}


def save_index(idx: dict):
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(idx, f)
    print(f"\n✓  Index saved → {INDEX_PATH}  ({len(idx['chunks'])} chunks total)")


def load_index() -> dict:
    if not Path(INDEX_PATH).exists():
        sys.exit(f"[ERROR] Index not found. Run: python rag.py ingest --corpus_dir ./corpus")
    with open(INDEX_PATH, "rb") as f:
        return pickle.load(f)


# ─── 3. RETRIEVAL ─────────────────────────────────────────────────────────────
def retrieve(question: str, idx: dict, k: int = TOP_K) -> list[dict]:
    q_vec  = idx["vectorizer"].transform([question])
    scores = cosine_similarity(q_vec, idx["matrix"])[0]
    top_i  = np.argsort(scores)[::-1][:k]
    results = []
    for i in top_i:
        c = dict(idx["chunks"][i])
        c["score"] = float(scores[i])
        results.append(c)
    return results


# ─── 4. UNANSWERABLE DETECTION ────────────────────────────────────────────────
def is_unanswerable(retrieved: list[dict]) -> bool:
    return (not retrieved) or retrieved[0]["score"] < UNANSWERABLE_THRESHOLD


# ─── 5. GENERATION (Gemini) ───────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a defence procurement policy assistant.
Answer questions ONLY using the provided context chunks.
Rules:
- Cite every factual claim with [doc_id] inline.
  Example: "The standstill period is 15 calendar days [doc_002]."
- If the context does not contain sufficient information, respond EXACTLY with:
  UNANSWERABLE: The corpus does not contain sufficient information to answer this question.
- Never use outside knowledge. Never hallucinate facts.
- Be concise and precise (3-5 sentences)."""


def generate_answer(question: str, retrieved: list[dict]) -> str:
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        return "[ERROR] GOOGLE_API_KEY not set. Run: export GOOGLE_API_KEY=your_key"

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
    except ImportError:
        return "[ERROR] Run: pip install google-genai"

    context_parts = [
        f"[{r['doc_id']}] (similarity={r['score']:.3f})\n{r['text']}"
        for r in retrieved
    ]
    context_str = "\n\n---\n\n".join(context_parts)
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context_str}\n\nQuestion: {question}"

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return response.text


# ─── 6. EVALUATION ────────────────────────────────────────────────────────────
def run_eval(questions_file: str, idx: dict):
    """
    Metrics:
      1. Avg top-1 retrieval cosine score  (retrieval quality proxy)
      2. Unanswerable detection rate
      3. Per-question answer printed for manual grounding check
    """
    lines = [
        l.strip() for l in Path(questions_file).read_text().splitlines()
        if l.strip()
    ]
    results = []

    print(f"\n{'='*65}")
    print("EVALUATION RUN")
    print(f"{'='*65}")

    for line in lines:
        # Strip "Q1:" prefix if present
        q = line.split(":", 1)[1].strip() if (line[0].isalpha() and ":" in line) else line
        retrieved  = retrieve(q, idx)
        unans_flag = is_unanswerable(retrieved)
        top_score  = retrieved[0]["score"] if retrieved else 0.0

        if unans_flag:
            answer = "UNANSWERABLE: Low retrieval confidence — question likely out of corpus scope."
        else:
            answer = generate_answer(q, retrieved)

        results.append({
            "q": q, "score": top_score,
            "unanswerable": unans_flag or answer.startswith("UNANSWERABLE"),
            "answer": answer,
        })

        tag = "⚠  UNANSWERABLE" if unans_flag else f"✓  score={top_score:.4f}"
        print(f"\nQ:  {q}")
        print(f"    {tag}")
        wrapped = textwrap.fill(answer, 80, subsequent_indent="    ")
        print(f"    A: {wrapped[:400]}")

    # ── Metric summary ──────────────────────────────────────────────────────
    n         = len(results)
    avg_score = np.mean([r["score"] for r in results])
    n_unans   = sum(r["unanswerable"] for r in results)

    print(f"\n{'='*65}")
    print("METRIC SUMMARY")
    print(f"  Questions evaluated         : {n}")
    print(f"  Avg top-1 retrieval score   : {avg_score:.4f}")
    print(f"  Answered                    : {n - n_unans}/{n}")
    print(f"  Flagged unanswerable        : {n_unans}/{n}")
    print(f"  Unanswerable threshold      : {UNANSWERABLE_THRESHOLD}")
    print(f"{'='*65}")
    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Plain-Vanilla RAG — Defence Procurement Policy QA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rag.py ingest --corpus_dir ./corpus
  python rag.py query  --question "What is the standstill period for contracts above £1 million?"
  python rag.py eval   --questions_file ./questions.txt
        """,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="Chunk + TF-IDF index the corpus")
    ing.add_argument("--corpus_dir", default="./corpus",
                     help="Directory containing .txt document files")

    qry = sub.add_parser("query", help="Answer a single question")
    qry.add_argument("--question", required=True)
    qry.add_argument("--k", type=int, default=TOP_K, help="Chunks to retrieve")

    evl = sub.add_parser("eval", help="Run evaluation over a questions file")
    evl.add_argument("--questions_file", default="./questions.txt")

    args = parser.parse_args()

    if args.cmd == "ingest":
        print(f"Loading corpus from: {args.corpus_dir}")
        chunks = load_corpus(args.corpus_dir)
        print(f"Total chunks built: {len(chunks)}")
        idx = build_index(chunks)
        save_index(idx)

    elif args.cmd == "query":
        idx       = load_index()
        retrieved = retrieve(args.question, idx, k=args.k)
        unans     = is_unanswerable(retrieved)

        print(f"\nQuestion : {args.question}")
        print(f"\nTop retrieved chunks:")
        for r in retrieved:
            bar = "█" * int(r["score"] * 100)
            print(f"  [{r['doc_id']}]  score={r['score']:.4f}  {bar}")
            print(f"    {r['text'][:120]}…\n")

        if unans:
            print("⚠  Retrieval score below threshold — question likely unanswerable from corpus.")
        print("Generating answer...\n")
        answer = generate_answer(args.question, retrieved)
        print(f"Answer:\n{answer}")

    elif args.cmd == "eval":
        idx = load_index()
        run_eval(args.questions_file, idx)


if __name__ == "__main__":
    main()
