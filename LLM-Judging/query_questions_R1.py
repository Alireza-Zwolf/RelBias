import os
import time
import argparse
import pandas as pd
import openai


"""
This code query DeepSeek R1 model from the original DeepSeek.com website by reading sensitive questions from a CSV file and saves the responses to another CSV file.


Usage Scripts:

python query_questions_R1.py \
  --questions ../csv_files/theme_questions_CS1.csv \
  --output ../csv_files/raw_responses/CaseStudy1_China/deepseekR1_CS1_responses.csv \
  --sleep 1


python query_questions_R1.py \
  --questions ../csv_files/theme_questions_CS2.csv \
  --output ../csv_files/raw_responses/CaseStudy2_US/deepseekR1_CS2_responses.csv \
  --sleep 1

"""




def load_censored_questions_from_CSV(questions_path="theme_questions.csv"):
    """
    Load the questions from a single CSV that includes Theme, Description, and Question columns.
    """
    return pd.read_csv(questions_path)

def query_deepseek(client, question):
    """
    Send a question to the DeepSeek R1 model and return the response.
    """
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ],
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API call failed: {e}")
        return f"Error: {e}"

def process_all_questions(merged_df, sleep_time=1):
    """
    Initialize the DeepSeek API client using the DEEPSEEK_API_KEY environment variable,
    then loop through all questions to collect responses.
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable not set.")

    client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    results = []
    for i, row in merged_df.iterrows():
        theme = row["Theme"]
        description = row["Description"]
        question = row["Question"]

        print(f"[{i+1}/{len(merged_df)}] Querying: {question[:60]}...")

        answer = query_deepseek(client, question)
        print(f"-----    Response: {answer[:60]}...")
        print("------------------------------------------------------")
        results.append({
            "Theme": theme,
            "Description": description,
            "Question": question,
            "DeepSeek_Response": answer
        })

        time.sleep(sleep_time)  # Avoid overloading the API

    return pd.DataFrame(results)

def save_responses(results_df, output_path="deepseek_responses.csv"):
    """
    Save the responses to a CSV file.
    """
    results_df.to_csv(output_path, index=False)
    print(f"Saved responses to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt DeepSeek R1 model with sensitive prompts.")
    parser.add_argument("--questions", type=str, default="theme_questions.csv", help="Path to the questions CSV file (must include Theme, Description, and Question columns)")
    parser.add_argument("--output", type=str, default="deepseek_responses.csv", help="Path to save the responses CSV")
    parser.add_argument("--sleep", type=float, default=1.0, help="Delay between queries in seconds")

    args = parser.parse_args()

    df = load_censored_questions_from_CSV(args.questions)
    results_df = process_all_questions(df, sleep_time=args.sleep)
    save_responses(results_df, args.output)



