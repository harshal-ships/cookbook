# Bedrock + Telcoflow

**Aria** is a B3networks **sales enquiry** voice agent: Telcoflow handles the call, **Amazon Nova Sonic 2** on Bedrock handles **native speech-to-speech** (no separate STT/TTS).
| Script | Description |
|--------|-------------|
| `aria_bedrock_agent.py` | Incoming call → answer → stream PCM to Nova → play Nova audio back; clears playback on barge-in |

Audio: Telcoflow **24 kHz** PCM → resampled to **16 kHz** for Nova input; Nova output is **24 kHz** back to the caller.

## Run

Python **3.12+** recommended for the experimental Bedrock runtime client.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # fill Telcoflow + IAM credentials
.venv/bin/python aria_bedrock_agent.py
```

## Environment

See `.env.example`: `WSS_API_KEY`, `WSS_CONNECTOR_UUID`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` (e.g. `us-east-1`). Use **IAM credentials** for `InvokeModelWithBidirectionalStream`.
