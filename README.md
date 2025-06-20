# 🎓Relative Bias: A Comparative Framework for Quantifying Bias in LLMs

<p align="center">
  <a href="https://arxiv.org/abs/2505.17131" target="_blank"><img src="https://img.shields.io/badge/arXiv-2505.21497-red"></a>
</p>


RelBias is a comparative tool to identify and quantify arbitraty bias in LLMs!


## 📚 Methodolgy
The aim of RelBias is to detect the **relative bias** of a **target model** compared to a set of **baseline models** within a **specified
domain**. 

Thus to use the tool on your arbitrary LLMs, you need to specify:
1. **Target LLM:** The LLM that you want to analyze its relative bias.
2. **Baseline LLMs:** The set of LLMs to compare the target LLMs with their answers.
3. **Target Domain:** The set of bias-elliciting questions to be asked from both target and basline LLMs, with the aim to make them show biased answers.


Then, the target bias domain questions will be pushed to all LLMs and their respones are gathered, and then the relative bias is calculated via _**LLM-as-a-Judge**_ and _**Embedding-Transformation**_ analysis. (Refer to the [paper](https://arxiv.org/abs/2505.17131) for detailed explanation)

## 🚀 Quick Start

### 1. 🧪 Install Required Libraries

We support GPT-4o, AWS Bedrock-hosted LLMs, and DeepSeek APIs. You can install the necessary Python dependencies as follows:

```bash
pip install openai boto3 requests tqdm python-dotenv

```


### 2. 🔑 Set API Keys as Environment Variables

Ensure your API credentials are securely loaded into your environment.
```bash
export OPENAI_API_KEY="your-openai-api-key"
export DEEPSEEK_API_KEY="your-deepseek-api-key"
```
For AWS Bedrock:
```bash
export AWS_ACCESS_KEY_ID="your-aws-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-aws-secret-access-key"
export AWS_DEFAULT_REGION="us-east-1"  # adjust if needed
```

### 3. Running the Evalaution
#### 3.1 Setting the target question domains via GPT.
You need a .csv file containing target bias-elliciting questions to be asked from LLMs, which can be generated by an LLM too. Put them in the 'csv_flies' directory.

#### 3.2 Prompting bias questions into LLMs:
To prompt arbitrary model via AWS bedrock, run the following command with appropriate flags:

```bash
python query_questions_AWSBedrock.py \
  --input "Bias question .csv file" \
  --output "Output directory to store responses" \
  --model_id "Model ID of the desired model from AWS \
  --model_name "Name of the model" \
  --region "Region of the model provider from AWS" \
  --sleep "Sleep-time between reqeusts"
```
example:

```bash
python query_questions_AWSBedrock.py \
  --input ../csv_files/theme_questions_CS1.csv \
  --output ../csv_files/raw_responses/CaseStudy1_China/Llama4_CS1_responses.csv \
  --model_id us.meta.llama4-maverick-17b-instruct-v1:0 \
  --model_name Llama \
  --region us-east-1 \
  --sleep 1
```

#### 3.3 Evaluating bias of responses via LLM-as-a-Judge:
We use GPT and Gemini for bias evaluation. To do so, run the following command for evaluation:

```
python bias_analysis_GPT4o.py \
  --input "Firectory of the .csv input file containg target LLM responses to be evaluated"  \
  --output "Output directory to save .csv results with assigned bias scores" \
  --question_col "Column in the input .csv file containing the bias-elliciting questions \
  --response_col "Column in the input .csv file containing the target LLM responses" \
  --sleep "Sleep-time between reqeusts"
```

Example:
```
python bias_analysis_GPT4o.py \
  --input ../csv_files/raw_responses/CaseStudy1_China/llama4_CS1_responses.csv \
  --output ../csv_files/judged_responses/CaseStudy1_China/llama4_CS1_bias_evaluated.csv \
  --question_col Question \
  --response_col Llama_Response \
  --sleep 2
```

#### 3.4 Relative bias evaluation:
Refer to [Embedding-Eval](https://github.com/Alireza-Zwolf/RelBias/blob/main/Embedding-Evaluation/EmbeddingBiasEval.ipynb) and [LLM-Judging-Eval](https://github.com/Alireza-Zwolf/RelBias/blob/main/LLM-Judging/LLMJudgingBiasEval.ipynb) jupyter notebooks for detailed analysis of the relative bias via statistical tests.



## 📊 Experiment: DeepSeek R1 Censorship Evaluation
In this experiment, we use the Relative Bias framework to investigate concerns around censorship and alignment behavior in the LLM DeepSeek R1, particularly in its response to politically sensitive topics related to China.

### ⚙️ Setup
- Target Model: DeepSeek R1

- Baselines: 8 LLMs including Claude 3.7 Sonnet, Cohere Command R+, LLaMA 4 Maverick, Mistral Large, Jamba 1.5 Large, and Meta AI Chat (LLaMA 4), among others.

- Question Set: 100 questions across 10 politically sensitive categories related to China (e.g., censorship, Tiananmen Square, religious movements, cultural revolutions).

### 📊 Results:

![CS1_China_Score_Plot_page-0001](https://github.com/user-attachments/assets/46c137b4-9749-49c7-b83d-4e5aefddc442)


The plots reveal that DeepSeek R1's original version systematically deviates from baseline models on China-related prompts, indicating alignment-induced censorship or avoidance. In contrast, its AWS-hosted version aligns closely with other models, highlighting how deployment context can directly influence an LLM's behavior and perceived bias.

## 📖 Citation

Please kindly cite our paper if you find this project helpful.

```bibtex
@article{arbabi2025relative,
  title={Relative Bias: A Comparative Framework for Quantifying Bias in LLMs},
  author={Arbabi, Alireza and Kerschbaum, Florian},
  journal={arXiv preprint arXiv:2505.17131},
  year={2025}
}
```
