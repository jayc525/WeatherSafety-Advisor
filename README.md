# Weather Advisory Support Bot

## What I built
A LangGraph chatbot that gives outdoor safety advice. It takes the user's question, fetches live weather from Open-Meteo, and matches it against safety policies (SOPs).

## Architecture
```text
User Question
      ↓
Intent Extraction (LLM)
      ↓
Location Resolution (Geocoding API)
      ↓
Live Weather Fetch (Open-Meteo API)
      ↓
SOP Matching (Python)
      ↓
Response Composition (LLM)
```

## Why LangGraph?
The assignment requires a real graph. LangGraph helps me make sure the steps happen in the right order (e.g., getting weather before checking rules) and lets me handle errors cleanly if an API fails.

## SOP Design
The safety rules are stored in `sops/sops.json`. 
I used JSON so the policies are separate from the code. Python reads this file and checks if the weather matches any rules. If multiple apply, it returns all of them.

## How to add an SOP
You can add a new SOP without changing the Python code:
1. Open `sops/sops.json`.
2. Add a new block:
```json
{
  "id": "SOP-013",
  "name": "Heavy Snow Warning",
  "category": "Travel",
  "severity": "critical",
  "applies_to_activities": ["driving", "cycling"],
  "applies_to_vulnerable_groups": [],
  "conditions": {
    "logic": "OR",
    "rules": [
      {"type": "threshold", "field": "snowfall", "operator": ">=", "value": 5}
    ]
  },
  "advice": "Heavy snowfall detected. Avoid all driving and cycling."
}
```
3. Save the file.

## LLM Responsibility
The LLM handles the user's language, while Python decides which SOP matches.
The LLM does NOT invent weather data or make up its own safety rules.

## Memory
The bot remembers context during the chat. If you ask "Is it safe to cycle in Bhopal?", and then ask "What about this evening?", it remembers you are talking about Bhopal and cycling.

## Failure Handling
- **API Failure:** If Open-Meteo fails, the graph stops and tells the user.
- **No SOP Match:** If the weather is fine, the bot says it has no specific guidance instead of making something up.

## Evaluation
I wrote tests in `eval/eval_suite.py` to check different scenarios using mocked weather data.

**Tests include:**
- Matching SOPs correctly
- Extreme weather handling
- What happens when no SOP matches
- API failures
- Prompt injection (making sure the LLM doesn't ignore the rules)
- Session memory

**Test Results:**
- 9/9 tests pass.

## Run Instructions

### 1. Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment
Rename `.env.example` to `.env` and add your API key:
```
GROQ_API_KEY=your_actual_api_key_here
```

### 3. Run the Chat App
```bash
streamlit run app.py
```

### 4. Run the Evaluation Suite
```bash
python eval/eval_suite.py
```
