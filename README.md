# AppleSupport AI Agent

## Setup Instructions (Under 15 mins)
1. Clone this repository.
2. Create a virtual environment: `python -m venv venv` and activate it.
3. Install dependencies: `pip install -r requirements.txt`
4. Add your Gemini API key to a `.env` file: `GEMINI_API_KEY=your_key_here`
5. Run the data filter: `python scripts/01_filter_data.py` (assuming you have `twcs.csv` in `data/raw/`).

## How to Evaluate
Run `python scripts/run_evaluation.py` to see the Agent classify intents and draft replies compared against a Baseline Agent. Note: Free Tier Gemini keys may experience 429/503 rate limits during the run.
