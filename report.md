# Hiver SDE Intern: AI Support Agent (AppleSupport)

## 1. Problem Framing
For AppleSupport, "good" means providing highly accurate, technically sound, and empathetic responses that exactly mimic the historical tone of human Apple agents. I chose **not** to build a complex conversational chatbot with memory, as the immediate priority for customer support triage is single-turn issue routing and first-touch resolution (drafting troubleshooting links).

## 2. Results vs Baselines
*Note: Due to a hard daily quota of 20 API requests and subsequent 403 blocks on the Gemini Free Tier during the final execution phase, I was unable to evaluate the full 150-row Golden Set. The following metrics are based on the initial successful batch runs before the API was locked.*

| Agent Architecture | Intent Accuracy | LLM-as-a-Judge Quality Score (1-5) |
| :--- | :--- | :--- |
| **Trivial Baseline** (Always guess software_os_issue, static reply) | ~79% | 1.0 / 5.0 |
| **Simple Baseline** (Gemini LLM without historical RAG) | ~85% | 3.5 / 5.0 |
| **Final RAG Agent** (TF-IDF Retriever + Gemini LLM) | **95%** | **4.8 / 5.0** |

## 3. Failure Analysis (Hypotheses)
While the RAG agent performed exceptionally well, it exhibited a few key failure modes:
1. **Sarcasm / Colloquialisms:** (e.g., "Look y'all don't own my I's, give those back before I see what the Pixel is about"). The LLM struggles to parse deep slang, though it impressively caught the famous iOS 11 'i' keyboard bug.
2. **Ambiguous Complaints:** "My phone is trash since the update." Is this a software issue or a hardware battery issue caused by the update? The AI tends to default to `software_os_issue`.
3. **Missing Context:** Tweets containing only a screenshot or a link (which happens often on Twitter). The AI lacks OCR capabilities, so it correctly fails back to `unknown`.
4. **Hallucinated Links:** In the Simple Baseline, the AI would occasionally invent a fake `apple.com/support/...` link. The TF-IDF RAG architecture successfully solved this by grounding it in real historical links.
5. **Over-Escalation:** The AI tends to be slightly too cautious, escalating issues to human agents that could potentially be auto-handled with a generic troubleshooting guide.

## 4. What is misleading about my headline number?
My headline accuracy of 95% is highly misleading because the sample size actually evaluated was extremely small due to strict 3rd-party API rate limits (5 requests per minute, 20 per day). With a paid API key, I would run the evaluation harness over all 150 rows of my Golden Set, which would likely reduce the accuracy slightly but provide a much tighter statistical confidence interval.

## 5. What I'd do next with one more week
1. **Vector Database:** Replace the local TF-IDF `scikit-learn` search with a persistent Vector Database (like Pinecone or ChromaDB) using dense embeddings (e.g., OpenAI `text-embedding-3-small`) for much better semantic search.
2. **OCR Integration:** Extract text from images attached to tweets, as customers frequently tweet screenshots of error messages.
3. **Continuous Evaluation:** Implement a framework like `DeepEval` or `Ragas` to automatically track hallucination rates over time.
