"""
RAG Baseline Evaluation Script

Evaluates the RAG system using:
1. Retrieval Hit Rate - Does the correct source PDF appear in retrieved chunks?
2. Keyword Match Score - What % of expected keywords appear in the generated answer?
3. Answer Correctness - Is keyword match >= 50%?

Usage:
    python evaluate.py

Prerequisites:
    - Run extract_keywords.py first to add keywords column to Eval_data_subset.csv
    - Start the FastAPI server: uvicorn main:app --reload
"""

import os
import pandas as pd
import requests
from tqdm import tqdm
from datetime import datetime

# Configuration
API_URL = "http://127.0.0.1:8000/query_with_context"
EVAL_DATA_PATH = "Eval_data_subset.csv"
RESULTS_PATH = "eval_results.csv"
CORRECTNESS_THRESHOLD = 0.5  # Keyword match >= 50% = correct


def check_retrieval_hit(sources: list, expected_pdf: str) -> bool:
    """Check if the expected PDF filename appears in any retrieved source."""
    if not sources:
        return False

    for source in sources:
        filename = source.get('filename', '')
        # Check if expected PDF name is in the filename (handle path variations)
        if expected_pdf and expected_pdf.lower() in filename.lower():
            return True
    return False


def calculate_keyword_match(generated_answer: str, keywords_str: str) -> float:
    """Calculate what percentage of expected keywords appear in the generated answer."""
    if not keywords_str or pd.isna(keywords_str):
        return 0.0

    # Parse keywords (comma-separated)
    keywords = [kw.strip().lower() for kw in keywords_str.split(',') if kw.strip()]
    if not keywords:
        return 0.0

    # Check how many keywords appear in the generated answer
    generated_lower = generated_answer.lower()
    matches = sum(1 for kw in keywords if kw in generated_lower)

    return matches / len(keywords)


def query_rag(query: str) -> dict:
    """Send a query to the RAG system and return the response."""
    try:
        response = requests.get(API_URL, params={"query": query}, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error querying RAG: {e}")
        return {"answer": "", "sources": []}


def main():
    # Load evaluation data
    df = pd.read_csv(EVAL_DATA_PATH)
    print(f"Loaded {len(df)} evaluation questions")

    # Check if keywords column exists
    if 'keywords' not in df.columns:
        print("ERROR: 'keywords' column not found. Run extract_keywords.py first.")
        return

    # Initialize results storage
    results = []
    total_hits = 0
    total_keyword_scores = 0
    total_correct = 0

    print("\n=== Starting RAG Evaluation ===\n")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        query = row['query']
        expected_pdf = row['pdf_filename']
        expected_answer = row['answer']
        keywords = row['keywords']

        # Query the RAG system
        response = query_rag(query)
        generated_answer = response.get('answer', '')
        sources = response.get('sources', [])

        # Calculate metrics
        hit = check_retrieval_hit(sources, expected_pdf)
        keyword_score = calculate_keyword_match(generated_answer, keywords)
        is_correct = keyword_score >= CORRECTNESS_THRESHOLD

        # Accumulate totals
        total_hits += int(hit)
        total_keyword_scores += keyword_score
        total_correct += int(is_correct)

        # Store detailed results
        results.append({
            'query_id': row['query_id'],
            'query': query,
            'expected_pdf': expected_pdf,
            'keywords': keywords,
            'generated_answer': generated_answer[:500],  # Truncate for CSV
            'retrieved_sources': ', '.join([s.get('filename', '') for s in sources]),
            'retrieval_hit': hit,
            'keyword_score': round(keyword_score, 3),
            'is_correct': is_correct
        })

    # Calculate aggregate metrics
    n = len(df)
    hit_rate = (total_hits / n) * 100
    avg_keyword_score = (total_keyword_scores / n) * 100
    correctness_rate = (total_correct / n) * 100

    # Print summary
    print("\n" + "=" * 50)
    print("=== RAG Baseline Evaluation Results ===")
    print("=" * 50)
    print(f"Questions Evaluated:    {n}")
    print(f"Retrieval Hit Rate:     {hit_rate:.1f}%")
    print(f"Avg Keyword Match:      {avg_keyword_score:.1f}%")
    print(f"Answer Correctness:     {correctness_rate:.1f}% (>={CORRECTNESS_THRESHOLD*100:.0f}% keyword match)")
    print("=" * 50)

    # Save detailed results
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nDetailed results saved to: {RESULTS_PATH}")

    # Show some examples of failures for analysis
    print("\n--- Sample Results (first 5 incorrect answers) ---")
    incorrect = [r for r in results if not r['is_correct']][:5]
    for r in incorrect:
        print(f"\nQuery: {r['query'][:80]}...")
        print(f"Expected keywords: {r['keywords']}")
        print(f"Keyword score: {r['keyword_score']}")
        print(f"Retrieval hit: {r['retrieval_hit']}")


if __name__ == "__main__":
    main()
