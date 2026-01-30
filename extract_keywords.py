"""
One-time script to extract keywords from expected answers in the evaluation dataset.
Uses OpenAI to extract 2-5 key terms/phrases from each answer.
Adds a 'keywords' column to Eval_data_subset.csv.
"""

import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# Load API key
config_path = "./openai_key.env"
load_dotenv(dotenv_path=config_path)
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

EXTRACT_PROMPT = """Extract the 2-5 most important keywords or key phrases from this answer.
These should be the core concepts that any correct answer must mention.
Return as comma-separated values, lowercase only.

Answer: {answer}

Keywords:"""


def extract_keywords(answer: str) -> str:
    """Extract keywords from an answer using OpenAI."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": EXTRACT_PROMPT.format(answer=answer)}
            ],
            temperature=0,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return ""


def main():
    # Load the evaluation data
    df = pd.read_csv("Eval_data_subset.csv")
    print(f"Loaded {len(df)} evaluation questions")

    # Extract keywords for each answer
    keywords_list = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting keywords"):
        keywords = extract_keywords(row['answer'])
        keywords_list.append(keywords)

    # Add keywords column
    df['keywords'] = keywords_list

    # Save updated CSV
    df.to_csv("Eval_data_subset.csv", index=False)
    print(f"Updated Eval_data_subset.csv with keywords column")

    # Show sample results
    print("\nSample keyword extractions:")
    for idx in range(min(5, len(df))):
        print(f"\nAnswer: {df.iloc[idx]['answer'][:100]}...")
        print(f"Keywords: {df.iloc[idx]['keywords']}")


if __name__ == "__main__":
    main()
