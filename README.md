# Generative AI Prompt Engineering Tasks

This notebook explores two practical prompt engineering techniques using Google Gemini as the LLM backend. Each task demonstrates a different prompting strategy applied to a real-world problem.

---

## Table of Contents

- [Overview](#overview)
- [Task 1 - Sentiment Analysis with Chain-of-Thought Prompting](#task-1---sentiment-analysis-with-chain-of-thought-prompting)
- [Task 2 - Hyperparameter Tuning Analysis with Tree-of-Thought Prompting](#task-2---hyperparameter-tuning-analysis-with-tree-of-thought-prompting)
- [Tech Stack](#tech-stack)
- [Setup and Requirements](#setup-and-requirements)

---

## Overview

| Task | Prompting Technique | Problem Type |
|------|---------------------|--------------|
| Task 1 | Chain-of-Thought (CoT) | Sentiment Classification |
| Task 2 | Tree-of-Thought (ToT) | Hyperparameter Selection |

Both tasks use **Gemini 2.5 Flash Lite** via the `google-genai` Python SDK, run in Google Colab.

---

## Task 1 - Sentiment Analysis with Chain-of-Thought Prompting

### What it does

Given a set of customer reviews scraped from an e-commerce platform (wireless earbuds), the notebook uses an LLM to:

- Classify each review as **Positive**, **Neutral**, or **Negative**
- Justify the classification step by step using Chain-of-Thought prompting

### Why Chain-of-Thought

Standard prompting asks the model to jump straight to an answer. Chain-of-Thought instructs the model to reason through the problem before giving a final label, which improves accuracy on subjective tasks like sentiment.

### Prompt Structure

The prompt instructs the model to follow four explicit steps:

1. Identify specific keywords in the review (e.g., "great battery", "poor fit")
2. Evaluate the overall tone of the reviewer
3. Classify as Positive, Neutral, or Negative
4. Write a one-sentence justification

### Sample Reviews Used

The notebook tests six reviews covering a range of sentiments, including praise for battery life and audio features, complaints about noise cancellation and fit, and a severe negative experience with a faulty product and poor customer service.

---

## Task 2 - Hyperparameter Tuning Analysis with Tree-of-Thought Prompting

### What it does

This task combines a classical ML pipeline with an LLM-guided analysis to choose the best hyperparameter configuration for an XGBoost regression model.

### Dataset

An AI jobs market dataset (`ai_jobs_market_2025_2026.csv`) is used. The target variable is `annual_salary_usd`. After dropping irrelevant columns and removing duplicates, the features are label-encoded and split into train/test sets.

### ML Pipeline

1. Data loading and inspection
2. Feature selection and cleaning
3. Exploratory data analysis (salary distribution, boxplots, correlation heatmap)
4. Label encoding of categorical columns
5. Train/test split (80/20)
6. XGBoost Regressor with 5-fold GridSearchCV over 16 configurations
7. Evaluation using RMSE, MAE, and R-squared on the test set

### Hyperparameter Grid

| Parameter | Values Tested |
|-----------|---------------|
| `n_estimators` | 100, 200 |
| `max_depth` | 3, 5 |
| `learning_rate` | 0.01, 0.1 |
| `subsample` | 0.8, 1.0 |

### Why Tree-of-Thought

Tree-of-Thought prompting asks the model to explore multiple reasoning paths before converging on a decision. This mirrors how a human expert would approach hyperparameter selection: examining performance, efficiency, and stability separately before making a final recommendation.

### LLM Analysis Steps (ToT Structure)

The prompt walks the model through four branches:

- **Path 1 (High Performer):** Find the top 2 configurations by CV score. Check if overfitting is present by examining the train/validation gap.
- **Path 2 (Efficiency):** Identify if any configuration is significantly faster while staying within 1-2% of the best score.
- **Path 3 (Stability):** Find the configuration with the smallest train/validation gap and check if it is still accurate enough to be useful.
- **Path 4 (Final Verdict):** Compare all three candidates and select one, with a justification grounded in the bias-variance tradeoff.

---

## Tech Stack

- Python 3
- Google Colab
- `google-genai` (Gemini API)
- `xgboost`
- `scikit-learn`
- `pandas`, `numpy`
- `matplotlib`, `seaborn`

---

## Setup and Requirements

1. Open the notebook in Google Colab.

2. Store your Gemini API key as a Colab secret named `KEY_`:

   ```
   Colab sidebar -> Secrets -> Add new secret -> Name: KEY_
   ```

3. Install the required package (already included in the notebook):

   ```bash
   pip install google-genai
   ```

4. For Task 2, upload the AI jobs dataset to your Google Drive at:

   ```
   MyDrive/ML_Datasets/ai_jobs_market_2025_2026.csv
   ```

5. Run all cells in order.
