import logging
import json
import time
import boto3
import pandas as pd
from botocore.exceptions import ClientError


Llama4_modelID = "meta.llama4-maverick-17b-instruct-v1:0"
Llama3_70B_ModelID = "us.meta.llama3-3-70b-instruct-v1:0"
claude37_Sonnet_ModelID = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"



"""
This code query the models via AWS Bedrock by reading sensitive questions from a CSV file and saves the responses to a CSV file.

Note that you need to setup AWS credentials and install the required libraries:
```bash
pip install boto3 pandas
```
You can set up your AWS credentials by running the following command in your terminal:
```bash
aws configure
```
This will prompt you to enter your AWS Access Key ID, Secret Access Key, region, and output format.
You can also set the AWS credentials in your environment variables:
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=your_region
```


Usage Scripts:

* Llama4-Maverick
python query_questions_AWSBedrock.py \
  --input ../csv_files/theme_questions_CS1.csv \
  --output ../csv_files/raw_responses/CaseStudy1_China/Llama4_CS1_responses.csv \
  --model_id us.meta.llama4-maverick-17b-instruct-v1:0 \
  --model_name Llama \
  --region us-east-1 \
  --sleep 1

python query_questions_AWSBedrock.py \
  --input ../csv_files/theme_questions_CS2.csv \
  --output ../csv_files/raw_responses/CaseStudy2_US/Llama4_CS2_responses.csv \
  --model_id us.meta.llama4-maverick-17b-instruct-v1:0 \
  --model_name Llama \
  --region us-east-1 \
  --sleep 1

python query_questions_AWSBedrock.py \
  --input ../csv_files/theme_questions_CS3_Meta.csv \
  --output ../csv_files/raw_responses/CaseStudy3_Meta/Llama4_CS3_responses.csv \
  --model_id us.meta.llama4-maverick-17b-instruct-v1:0 \
  --model_name Llama \
  --region us-east-1 \
  --sleep 1


* Llama3.3-70B
python query_questions_AWSBedrock.py \
  --input ../csv_files/theme_questions_CS1.csv \
  --output ../csv_files/raw_responses/CaseStudy1_China/Llama3_CS1_responses.csv \
  --model_id us.meta.llama3-3-70b-instruct-v1:0 \
  --model_name Llama \
  --region us-east-1 \
  --sleep 1

python query_questions_AWSBedrock.py \
  --input ../csv_files/theme_questions_CS2.csv \
  --output ../csv_files/raw_responses/CaseStudy2_US/Llama3_CS2_responses.csv \
  --model_id us.meta.llama3-3-70b-instruct-v1:0 \
  --model_name Llama \
  --region us-east-1 \
  --sleep 1

python query_questions_AWSBedrock.py \
  --input ../csv_files/theme_questions_CS3_Meta.csv \
  --output ../csv_files/raw_responses/CaseStudy3_Meta/Llama3_CS3_responses.csv \
  --model_id us.meta.llama3-3-70b-instruct-v1:0 \
  --model_name Llama \
  --region us-east-1 \
  --sleep 1

  
* Claude3.7-Sonnet 
python query_questions_AWSBedrock.py \
  --input ../csv_files/theme_questions_CS1.csv \
  --output ../csv_files/raw_responses/CaseStudy1_China/ClaudeSonnet_CS1_responses.csv \
  --model_id us.anthropic.claude-3-7-sonnet-20250219-v1:0 \
  --model_name Claude \
  --region us-east-1 \
  --sleep 2

python query_questions_AWSBedrock.py \
  --input ../csv_files/theme_questions_CS2.csv \
  --output ../csv_files/raw_responses/CaseStudy2_US/ClaudeSonnet_CS2_responses.csv \
  --model_id us.anthropic.claude-3-7-sonnet-20250219-v1:0 \
  --model_name Claude \
  --region us-east-1 \
  --sleep 2

python query_questions_AWSBedrock.py \
  --input ../csv_files/theme_questions_CS3_Meta.csv \
  --output ../csv_files/raw_responses/CaseStudy3_Meta/ClaudeSonnet_CS3_responses.csv \
  --model_id us.anthropic.claude-3-7-sonnet-20250219-v1:0 \
  --model_name Claude \
  --region us-east-1 \
  --sleep 4



* Cohere Command R+
python query_questions_AWSBedrock.py \
  --input ../csv_files/theme_questions_CS1.csv \
  --output ../csv_files/raw_responses/CaseStudy1_China/CohereCommandR+_CS1_responses.csv \
  --model_id cohere.command-r-plus-v1:0 \
  --model_name CohereCommandR+ \
  --region us-east-1 \
  --sleep 1


python query_questions_AWSBedrock.py \
  --input ../csv_files/theme_questions_CS2.csv \
  --output ../csv_files/raw_responses/CaseStudy2_US/CohereCommandR+_CS2_responses.csv \
  --model_id cohere.command-r-plus-v1:0 \
  --model_name CohereCommandR+ \
  --region us-east-1 \
  --sleep 1

python query_questions_AWSBedrock.py \
  --input ../csv_files/theme_questions_CS3_Meta.csv \
  --output ../csv_files/raw_responses/CaseStudy3_Meta/CohereCommandR+_CS3_responses.csv \
  --model_id cohere.command-r-plus-v1:0 \
  --model_name CohereCommandR+ \
  --region us-east-1 \
  --sleep 1


* DeepSeek R1 AWS
python query_questions_AWSBedrock.py \
  --input ../csv_files/theme_questions_CS2.csv \
  --output ../csv_files/raw_responses/CaseStudy2_US/deepseekAWS_CS2_responses.csv \
  --model_id us.deepseek.r1-v1:0\
  --model_name DeepSeekAWS \
  --region us-east-1 \
  --sleep 1

python query_questions_AWSBedrock.py \
  --input ../csv_files/theme_questions_CS3_Meta.csv \
  --output ../csv_files/raw_responses/CaseStudy3_Meta/deepseekAWS_CS3_responses.csv \
  --model_id us.deepseek.r1-v1:0\
  --model_name DeepSeekAWS \
  --region us-east-1 \
  --sleep 1


* Mistral 
python query_questions_AWSBedrock.py \
  --input ../csv_files/theme_questions_CS3_Meta.csv \
  --output ../csv_files/raw_responses/CaseStudy3_Meta/MistralLarge_CS3_responses.csv \
  --model_id mistral.mistral-large-2402-v1:0\
  --model_name MistralLarge \
  --region us-east-1 \
  --sleep 1

python query_questions_AWSBedrock.py \
  --input ../csv_files/theme_questions_CS2.csv \
  --output ../csv_files/raw_responses/CaseStudy2_US/MistralLarge_CS2_responses.csv \
  --model_id mistral.mistral-large-2402-v1:0\
  --model_name MistralLarge \
  --region us-east-1 \
  --sleep 1

python query_questions_AWSBedrock.py \
  --input ../csv_files/theme_questions_CS1.csv \
  --output ../csv_files/raw_responses/CaseStudy1_China/MistralLarge_CS1_responses.csv \
  --model_id mistral.mistral-large-2402-v1:0\
  --model_name MistralLarge \
  --region us-east-1 \
  --sleep 1

  
* Jamba 1.5 Large
python query_questions_AWSBedrock.py \
  --input ../csv_files/theme_questions_CS3_Meta.csv \
  --output ../csv_files/raw_responses/CaseStudy3_Meta/Jamba_responses_CS3.csv \
  --model_id ai21.jamba-1-5-large-v1:0\
  --model_name Jamba \
  --region us-east-1 \
  --sleep 1

python query_questions_AWSBedrock.py \
  --input ../csv_files/theme_questions_CS2.csv \
  --output ../csv_files/raw_responses/CaseStudy2_US/Jamba_responses_CS2.csv \
  --model_id ai21.jamba-1-5-large-v1:0\
  --model_name Jamba \
  --region us-east-1 \
  --sleep 1


python query_questions_AWSBedrock.py \
  --input ../csv_files/theme_questions_CS1.csv \
  --output ../csv_files/raw_responses/CaseStudy1_China/Jamba_responses_CS1.csv \
  --model_id ai21.jamba-1-5-large-v1:0\
  --model_name Jamba \
  --region us-east-1 \
  --sleep 1




"""






logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def query_model_bedrock(modelID, prompt, region_name="us-east-1", whole_response=False):
    """
    Query a model hosted on AWS Bedrock with a user prompt.
    """
    brt = boto3.client("bedrock-runtime", region_name=region_name)
    conversation = [
        {
            "role": "user",
            "content": [{"text": prompt}],
        }
    ]

    try:
        response = brt.converse(
            modelId=modelID,
            messages=conversation,
            inferenceConfig={
                "maxTokens": 2000,
                "temperature": 0.7,
                "topP": 0.9,
            },
        )
        return response if whole_response else response["output"]["message"]["content"][0]["text"]

    except (ClientError, Exception) as e:
        logger.error(f"ERROR: Can't invoke '{modelID}'. Reason: {e}")
        return None

def process_all_questions_via_bedrock(merged_df, modelID, model_name, region_name="us-east-1", sleep_time=1):
    """
    Loop through all questions and get responses from the target model on AWS Bedrock.
    """
    results = []
    for i, row in merged_df.iterrows():
        theme = row["Theme"]
        description = row["Description"]
        question = row["Question"]

        print(f"[{i+1}/{len(merged_df)}] Querying: {question[:60]}...")

        answer = query_model_bedrock(modelID, question, region_name=region_name)
        results.append({
            "Theme": theme,
            "Description": description,
            "Question": question,
            f"{model_name}_Response": answer
        })

        time.sleep(sleep_time)  # Be nice to the API

    return pd.DataFrame(results)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Query a model hosted on AWS Bedrock with a list of questions.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input CSV file with Theme, Description, and Question")
    parser.add_argument("--output", type=str, required=True, help="Path to save the responses")
    parser.add_argument("--model_id", type=str, required=True, help="The model ID to use on AWS Bedrock")
    parser.add_argument("--model_name", type=str, required=True, help="A short readable name for the model (e.g., Llama, Claude)")
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region name")
    parser.add_argument("--sleep", type=float, default=1.0, help="Delay between queries in seconds")

    args = parser.parse_args()

    df = pd.read_csv(args.input)
    results_df = process_all_questions_via_bedrock(df, args.model_id, args.model_name, region_name=args.region, sleep_time=args.sleep)
    results_df.to_csv(args.output, index=False)
    print(f"\n✅ Saved Bedrock model responses to {args.output}")
