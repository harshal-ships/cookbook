# Deploying HealthFirst ADK Agents

This subproject is designed for local execution first. When you outgrow a laptop or VM, use Google Cloud Agent Runtime or Cloud Run.

References:

- [Agent Development Kit overview](https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/adk)
- ADK bidirectional streaming with `Runner.run_live()` and `LiveRequestQueue`

## What you are deploying

| Process | Purpose |
| --- | --- |
| `python run.py` with `AGENT_MODE=booking` | Inbound booking calls |
| `python run.py` with `AGENT_MODE=reminder` | Hourly reminders + optional reminder calls |
| `python run.py` with `AGENT_MODE=both` | Booking and reminder together |

Telcoflow must be able to reach your process over WebSocket. That usually means:

- a VM with a public egress path, or
- a container with stable outbound connectivity, or
- your own VPN/tunnel if Telcoflow requires it

This repo does not include Render-specific boot logic.

## Recommended production layout

```text
Cloud Run service OR GCE VM
  ├─ booking worker (Telcoflow WSS client)
  ├─ reminder worker (optional second service)
  ├─ persistent disk or Cloud Storage for bookings.json
  └─ Secret Manager for API keys and calendar JSON
```

Keep booking and reminder in separate Cloud Run services if you want independent scaling. Use `AGENT_MODE=both` only for small demos.

## Environment variables

Store these in Secret Manager or your platform's secret store:

- `WSS_API_KEY`
- `WSS_CONNECTOR_UUID`
- `GOOGLE_API_KEY`
- `GOOGLE_CALENDAR_CREDENTIALS_JSON`
- `GOOGLE_CALENDAR_ID`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CONFIRM_CHAT_ID`

Non-secret config can stay as plain env vars:

- `AGENT_MODE`
- `CLINIC_TIMEZONE`
- `APPOINTMENT_DURATION_MINUTES`
- `REMINDER_CHECK_INTERVAL_SECONDS`

## Cloud Run sketch

Cloud Run is a good fit when you want a managed container without Render/OpenClaw.

1. Build a Docker image from `adk_healthfirst/Dockerfile` (create one if needed).
2. Mount or sync `bookings.json` using Cloud Storage FUSE or write to Firestore later.
3. Inject secrets from Secret Manager.
4. Run one service per mode:
   - `AGENT_MODE=booking`
   - `AGENT_MODE=reminder`

Important: Telcoflow is a long-lived WebSocket client. Cloud Run can work, but configure:

- minimum instances = 1
- CPU always allocated
- no request-based autoscaling to zero if you need continuous call handling

## Agent Runtime sketch

If you later move the ADK agent itself onto [Gemini Enterprise Agent Platform Agent Runtime](https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/adk):

1. Keep Telcoflow audio bridging in a small sidecar/worker if Runtime does not expose your phone connector directly.
2. Register the ADK agent with tools:
   - `check_appointment_availability`
   - `find_next_available_slots`
3. Use Runtime sessions for voice turns and keep Calendar writes in deterministic Python services.

Practical split:

- **Runtime ADK agent** = reasoning + live tools
- **Worker service** = Telcoflow bridge + Calendar API + bookings.json

That mirrors this repo's current separation between voice and post-call booking logic.

## Observability

Log these on every call:

- transcript lines
- tool calls from ADK (`check_appointment_availability`, `find_next_available_slots`)
- post-call JSON status
- created `calendar_event_id`

For production, ship logs to Cloud Logging and alert on:

- repeated `needs_human_review`
- calendar API 403/404 errors
- Telcoflow reconnect loops

## Security checklist

- Share the target Google Calendar only with the service account email
- Never commit credentials JSON
- Restrict Telegram bot chat IDs
- Rotate `GOOGLE_API_KEY` and Telcoflow keys independently

## Local vs production difference

| Concern | Local | Production |
| --- | --- | --- |
| Secrets | `.env` | Secret Manager |
| Bookings store | `./bookings.json` | mounted disk or database |
| Reminders | Telegram message | Telegram + optional outbound voice |
| Voice stack | ADK `run_live()` | same, optionally hosted on Agent Runtime |
