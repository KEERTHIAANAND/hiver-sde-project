# Decision Log

1. **Choosing 5 Broad Intents:** I chose to group intents by high-level departments (`software`, `hardware`, `account`, `general`, `unknown`) rather than granular issues (like `cracked_screen`). This mirrors real-world support routing, which routes to teams rather than specific sub-problems.
2. **The 'Unknown' Intent:** Added a mandatory `unknown` bucket to prevent the AI from guessing on noisy Twitter data (gibberish, single links, or completely unrelated complaints).
3. **No Aggressive Data Cleaning:** In traditional ML, I would remove emojis and slang. With an LLM, I intentionally kept the raw text so the AI could understand customer emotion and match Apple's exact historical formatting.
4. **TF-IDF over BERT Embeddings:** For the Retriever, I used `scikit-learn`'s TF-IDF. It runs instantly in memory on a local machine for 100k+ rows without the overhead or cost of generating dense embeddings for a small prototype.
5. **LLM over Fine-Tuned Classifier:** I chose Gemini 3.6 Flash over training a custom BERT classifier because the assignment required the agent to *draft a reply*. Using a single LLM prompt handles intent classification, action routing, and text generation all at once.
6. **JSON Structured Output:** Enforced strict JSON output from the LLM prompt to ensure the agent's decisions could be programmatically parsed by backend routing systems.
7. **RAG Architecture (Retrieval-Augmented Generation):** Rather than letting the LLM hallucinate answers, I fed it the top 2 historical Apple replies to similar issues to ground its tone and provide real troubleshooting links.
8. **Dual-Metric Evaluation:** Accuracy alone isn't enough for generative text. I implemented an LLM-as-a-judge metric to score the drafted replies from 1-5 on tone and helpfulness.
9. **Bulletproof Retry Logic:** Added `try/except` backoff loops inside the evaluation harness to gracefully handle `429` and `503` API timeouts without crashing the script.
10. **Pre-filtering the Dataset:** Wrote a pandas script (`01_filter_data.py`) to extract only complete `(Inbound, Outbound)` conversation pairs specifically for AppleSupport, reducing 3 million messy rows to 106k high-quality training pairs.
