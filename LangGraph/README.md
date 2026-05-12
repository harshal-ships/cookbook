# LangGraph + Telcoflow

Voice agents that use **Telcoflow** for phone audio and **LangGraph** for per-call state (history, tools, when to end the call). Each call is keyed by `call.call_id`.

| Script | Model | Role |
|--------|--------|------|
| `nova_gemini_agent.py` | Google Gemini Live | B3networks customer care **Nova** — tools + LangGraph |
| `nova_bedrock_agent.py` | Amazon Nova Sonic 2 (Bedrock) | Same pattern as Gemini; **24 kHz** Telcoflow audio downsampled to **16 kHz** for Nova input |

**Helpers:** `test_nova_sonic.py` (Bedrock text smoke test), `test_sonic_apikey.py` (bidirectional stream + API key check).

## Run

Use **Python 3.12+**, a virtualenv, and install from `requirements.txt` (Telcoflow from TestPyPI + Bedrock client from GitHub — see file comments).

```bash
source .venv-test/bin/activate   # or your venv
pip install -r requirements.txt
cp .env.example .env            # fill in keys
python nova_gemini_agent.py
# or
python nova_bedrock_agent.py
```

## Environment

See `.env.example`: Telcoflow (`WSS_*`), Gemini (`GOOGLE_API_KEY`) for the Gemini agent, and **IAM** AWS keys for Nova Sonic (bidirectional streaming does not use Bedrock API keys per AWS docs).
