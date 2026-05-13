# LangGraph + Telcoflow

## Overview

**Nova** is a B3networks **customer care** voice agent: **Telcoflow** carries audio; **Gemini Live** or **Nova Sonic 2** does speech-to-speech. **LangGraph** keeps **per-call state** (history, tool results, turn count, whether to end the call) and runs **when the model emits tool calls or completes a turn**—not on every PCM chunk. Use this when you need **stateful graphs, tool routing, and checkpointing** keyed by `call.call_id`.

| Script | Model | Role |
|--------|--------|------|
| `nova_gemini_agent.py` | Google Gemini Live | Nova + **tools** + LangGraph |
| `nova_bedrock_agent.py` | Amazon Nova Sonic 2 (Bedrock) | Same idea; **24 kHz** Telcoflow → **16 kHz** Nova input |



## Run

Use **Python 3.12+** and `requirements.txt` (TestPyPI for `telcoflow-sdk`, GitHub for Bedrock streaming client).

```bash
source .venv-test/bin/activate   # or your venv
pip install -r requirements.txt
cp .env.example .env            # fill in keys
python nova_gemini_agent.py
# or
python nova_bedrock_agent.py
```

## Environment

See `.env.example`: Telcoflow (`WSS_*`), Gemini (`GOOGLE_API_KEY`) for the Gemini agent, and **IAM** AWS keys for Nova Sonic (Bedrock API keys do not support bidirectional streaming per AWS docs).
