# AI Receptionist with Database Lookup

Telcoflow inbound calls + **Amazon Nova 2 Sonic** + local CRM lookup + optional **human handoff**.

Implements the [AI Receptionist with Database Lookup](https://docs.agentao.com/use-cases/database-lookup) pattern.

## Flow

```text
Inbound call (Telcoflow)
        │
        ▼
  answer() + lookup call.caller_number in customers.json
        │
        ├─ Known customer → personalized Nova prompt + open tickets
        └─ Unknown caller → create lead in leads.json
        │
        ▼
  Nova 2 Sonic conversation
        │
        ├─ AI resolves intent → record_intent, continue talking
        └─ Caller needs human → transfer_to_human
                │
                ▼
          connect() → callee rings → close() → AI leaves
```

**State flow (with handoff):** PENDING → ANSWERED → CONNECTED → DISCONNECTED

## CRM backends

Set `CRM_BACKEND` in `.env`:

| Value | Backend |
| --- | --- |
| `json` (default) | Local `customers.json` + `leads.json` — demo/offline |
| `pipedrive` | Live [Pipedrive API](https://developers.pipedrive.com/docs/api/v1/Persons) |

### JSON demo (`CRM_BACKEND=json`)

| Phone | Customer |
| --- | --- |
| `+919876543210` | Priya Sharma (2 open tickets) |
| `+14155550123` | James Lee (1 open ticket) |

### Pipedrive (`CRM_BACKEND=pipedrive`)

```bash
CRM_BACKEND=pipedrive
PIPEDRIVE_API_TOKEN=your_api_token
PIPEDRIVE_COMPANY_DOMAIN=yourcompany   # yourcompany.pipedrive.com
```

On each call the agent:

1. **Searches persons** by phone (`GET /persons/search`)
2. **Known caller** → loads open **deals** as tickets (`GET /persons/{id}/deals`)
3. **Unknown caller** → creates **Person** + **Lead** in Pipedrive
4. **`update_lead_name`** → updates the Person name in Pipedrive

Get your API token: Pipedrive → Settings → Personal preferences → API.

Every call is still logged locally in `reception_calls.json`.

## Tools (Nova Sonic)

| Tool | Purpose |
| --- | --- |
| `get_open_tickets` | Read open tickets for a customer |
| `record_intent` | Log billing / sales / support / general |
| `update_lead_name` | Save name for new leads |
| `transfer_to_human` | Trigger `connect()` + `close()` handoff |

## Setup

```bash
cd ai_receptionist
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python receptionist_agent.py
```

### Prerequisites

- Telcoflow connector (`WSS_API_KEY`, `WSS_CONNECTOR_UUID`)
- **Python 3.12+** (`aws-sdk-bedrock-runtime`)
- AWS credentials + **Nova 2 Sonic** enabled in Bedrock (`us-east-1` typical)
- Telcoflow connector must route to a real callee for handoff to work

## Test scenarios

1. **Known caller** — call from a number in `customers.json` → hear personalized greeting mentioning tickets.
2. **New caller** — unknown number → lead created, generic welcome.
3. **Human handoff** — say *"I need to speak to someone"* → AI says connecting → call rings through.

## Files

- `receptionist_agent.py` — main agent + CRM
- `nova_sonic_bridge.py` — Telcoflow ↔ Nova audio + handoff signal
- `customers.json` — CRM + tickets
- `leads.json` — new callers
- `reception_calls.json` — call audit log
