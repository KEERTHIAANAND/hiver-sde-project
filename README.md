# AppleSupport AI Agent

An end-to-end AI customer support agent built to triage, classify, and draft historically grounded replies for inbound `@AppleSupport` tweets.

## 🧠 Technical Architecture & Working Flow

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline to prevent LLM hallucinations and maintain historical brand tone. 

1. **Data Pre-processing (`scripts/01_filter_data.py`)**
   - Ingests ~3M rows of raw, noisy Twitter data.
   - Filters out irrelevant brands and pairs `(inbound_customer_tweet, outbound_agent_reply)`.
   - Results in a high-quality historical dataset of ~106k conversational pairs.

2. **The RAG Retriever (`src/retriever.py`)**
   - Converts all historical customer queries into vector embeddings using fast, in-memory **TF-IDF (Term Frequency-Inverse Document Frequency)**.
   - When a new tweet arrives, it calculates cosine similarity to pull the top 2 most mathematically similar past conversations and retrieves the exact links/solutions the human agents provided.

3. **The Generative Agent (`src/agent.py`)**
   - Uses the **Gemini LLM** to analyze the incoming tweet alongside the historical context fetched by the Retriever.
   - Outputs a strict JSON payload that classifies the query into 1 of 5 broad intents, drafts a highly-grounded reply, and routes the ticket to either `auto-handle` or `escalate` (with reasoning).

4. **The Evaluation Harness (`scripts/run_evaluation.py`)**
   - Runs the test set through a "Simple Baseline" (No RAG) and the "Final RAG Agent" to compare performance.
   - Employs an **LLM-as-a-judge** metric to evaluate the drafted replies on Tone, Relevance, and Safety (scoring 1-5).
   - *Note: Includes robust exponential backoff `try/except` loops to seamlessly handle free-tier API `429/503` rate limits.*

---

## 🚀 Setup Instructions (Under 15 mins)
1. Clone this repository.
2. Create a virtual environment: `python -m venv venv` and activate it.
3. Install dependencies: `pip install -r requirements.txt`
4. Add your Gemini API key to a `.env` file in the root directory: `GEMINI_API_KEY=your_key_here`
5. Run the data filter: `python scripts/01_filter_data.py` (assuming you have `twcs.csv` in `data/raw/`).

## 📊 How to Evaluate
Run the following command to see the Agent classify intents and draft replies compared against a Baseline Agent on the hand-labeled Golden Set:
```bash
python scripts/run_evaluation.py
