import pandas as pd
import time
import argparse
from meta_ai_api import MetaAI  # pip install meta-ai-api



"""
This code query Meta AI by reading sensitive questions from a CSV file and saves the responses to a CSV file.
Usage:

python query_meta.py \
    --input ../csv_files/theme_questions_CS3_Meta.csv \
    --output ../csv_files/raw_responses/CaseStudy3_Meta/MetaAI_CS3_responses.csv

"""


def query_meta_ai(prompt):
    ai = MetaAI()
    response = ai.prompt(message=prompt)
    return response["message"]

def main(input_csv, output_csv):
    # Load input CSV
    df = pd.read_csv(input_csv)

    # Ensure 'Question' column exists
    if 'Question' not in df.columns:
        raise ValueError("Input CSV must have a 'Question' column")

    # Initialize MetaResponse column
    df["MetaResponse"] = ""

    # Query Meta AI for each question
    for idx, row in df.iterrows():
        question = row["Question"]
        print(f"Processing Q{idx + 1}: {question}")
        try:
            response = query_meta_ai(question)
        except Exception as e:
            response = f"Error: {str(e)}"
        print(f"Response: {response[:200]} ...")
        print("------------------------------------------------------")
        df.at[idx, "MetaResponse"] = response
        time.sleep(2)  # Avoid rate limiting

    # Save results
    df.to_csv(output_csv, index=False)
    print(f"Saved responses to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query Meta AI for sensitive questions.")
    parser.add_argument("--input", required=True, help="Path to the input CSV file.")
    parser.add_argument("--output", required=True, help="Path to save the output CSV with responses.")
    args = parser.parse_args()

    main(args.input, args.output)
