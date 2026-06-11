
# After-Hours Voicemail Agent

Telcoflow inbound calls with **time-of-day routing**: connect to staff during business hours, or take a voicemail after hours using **Gemini Live**, then notify the team on **Telegram**.

Inspired by [After-hours Voicemail and Notification](https://docs.agentao.com/use-cases/after-hours-voicemail).

## Flow

```text
Inbound call (Telcoflow)
        │
        ├─ Business hours (Mon–Fri, configurable)
        │     connect() → callee rings → close() → agent leaves
        │
        └─ After hours
              answer() → Gemini Live voicemail script
              → transcript saved to voicemails.json
              → Telegram message to team
              → disconnect()
```

| Layer | Role |
| --- | --- |
| **Telcoflow SDK** | Phone audio in/out, `connect` / `answer` / `close` |
| **Gemini Live** | After-hours greeting, listen, transcribe caller message |
| **voicemails.json** | Local ledger of recorded messages |
| **Telegram Bot API** | Team notification (no Slack) |

## Setup

```bash
cd after_hours_voicemail
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with WSS_*, GOOGLE_API_KEY, TELEGRAM_*
python voicemail_agent.py
```

### Telegram bot

1. Create a bot with [@BotFather](https://t.me/BotFather) → copy `TELEGRAM_BOT_TOKEN`.
2. Start a chat with the bot (or add it to a group).
3. Get `TELEGRAM_CHAT_ID`:
   - DM: message the bot, then open `https://api.telegram.org/bot<TOKEN>/getUpdates`
   - Group: add bot to group, send a message, read `chat.id` from `getUpdates`
4. Set both values in `.env`.

### Telcoflow connector

Point your connector at this deployment. The agent uses outbound WSS with `WSS_API_KEY` and `WSS_CONNECTOR_UUID` (sandbox mode in code; switch to `TelcoflowClientConfig.production(...)` for prod).

## Environment

| Variable | Purpose |
| --- | --- |
| `WSS_API_KEY`, `WSS_CONNECTOR_UUID` | Telcoflow connector |
| `GOOGLE_API_KEY` | Gemini Live |
| `GEMINI_MODEL` | Live audio model (default native-audio preview) |
| `BUSINESS_TIMEZONE` | e.g. `Asia/Kolkata` |
| `BUSINESS_OPEN_HOUR`, `BUSINESS_CLOSE_HOUR` | 24h integers, weekdays only |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | Team alerts |
| `VOICEMAILS_PATH` | JSON store path |

## Output

Each after-hours call appends to `voicemails.json`:

```json
{
  "id": "uuid",
  "call_id": "telcoflow-call-id",
  "caller_number": "+1...",
  "recorded_at": "2026-05-28T21:30:00+05:30",
  "transcript": "caller message text",
  "telegram_notified": true
}
```

## Notes

- Business hours = **Monday–Friday** only; weekends always go to voicemail.
- Raw audio is not stored (transcript only). Add WAV persistence later if needed.
- For production mTLS, replace `make_telcoflow_config()` with `TelcoflowClientConfig.production(cert_path=..., key_path=...)`.
