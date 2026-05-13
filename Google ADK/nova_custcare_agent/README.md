## Overview

Nova is a fully autonomous customer care voice agent for **B3Networks** built with the **Google ADK** (Agent Development Kit) and the **Telcoflow SDK**. Nova answers every call, identifies the caller, and resolves their issue without ever escalating to a human.

This is the first agent in the repo to use the **ADK pipeline** (`Agent → Runner → run_live()`) instead of the raw GenAI Live API. ADK handles the tool-execution loop automatically — no manual `tool_call` / `send_tool_response` dispatch.

## What Nova Can Do

| Capability | Tool | What happens |
|---|---|---|
| Account lookup | `look_up_customer` | Finds customer by phone, returns name, plan, balance |
| Ticket lookup | `get_ticket_details` | Checks a specific ticket by ID |
| Ticket list | `list_customer_tickets` | Shows all tickets for the caller |
| Ticket creation | `create_ticket` | Opens a new ticket with subject, description, priority |
| Billing inquiry | `get_billing_summary` | Returns invoices, amounts, due dates, payment status |
| Product info | `get_product_info` | Explains B3Networks products and services |
| FAQ / troubleshooting | `search_faq` | Searches the knowledge base for answers |
| Contact update | `update_customer_email` | Changes the email on file |
| End call | `end_call` | Wraps up the conversation and disconnects |

## Architecture

```
Caller ←PCM→ Telcoflow ←PCM→ agent.py ←LiveRequestQueue→ Google ADK Runner
                                 │                              │
                                 │         ADK auto-executes:   │
                                 │         ├── look_up_customer  → DB read
                                 │         ├── get_ticket_details → DB read
                                 │         ├── list_customer_tickets → DB read
                                 │         ├── create_ticket     → DB write
                                 │         ├── get_billing_summary → DB read
                                 │         ├── get_product_info  → DB read
                                 │         ├── search_faq        → DB read
                                 │         ├── update_customer_email → DB write
                                 │         └── end_call          → call.disconnect()
                                 │
                                 └── Post-call: log interaction to DB
```

### ADK vs GenAI (how Nova differs from other agents)

| | Other agents (raw GenAI) | Nova (Google ADK) |
|---|---|---|
| **Model session** | `gemini.aio.live.connect()` | `Agent` + `Runner` + `run_live()` |
| **Tool declarations** | Manual JSON `function_declarations` | Auto-generated from Python type hints |
| **Tool dispatch** | Manual `response.tool_call` loop + `send_tool_response` | ADK Runner handles it transparently |
| **Audio bridge** | `session.send_realtime_input()` / `session.receive()` | `LiveRequestQueue.send_realtime()` / `runner.run_live()` events |
| **Session memory** | None (stateless) | `InMemorySessionService` (swappable for persistent) |

## State Flow

```
PENDING → ANSWERED → DISCONNECTED
```

Nova never routes to a human. Every call ends with `disconnect()`.

## Project Structure

```
nova_customer_care/
├── agent.py          # Entrypoint — Telcoflow + ADK bridge, per-call session
├── tools.py          # Tool closure factory (9 tools, auto-registered by ADK)
├── database.py       # aiosqlite — customers, tickets, products, billing, FAQ
├── config.py         # Env-based configuration
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

```bash
cd nova_customer_care
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://test.pypi.org/simple/
cp .env.example .env
# Fill in: GEMINI_API_KEY, WSS_API_KEY, WSS_CONNECTOR_UUID
```

## Run

```bash
python agent.py
```

```
2026-04-15 10:00:00  nova                      INFO   Nova [B3Networks] is live — fully autonomous customer care — waiting for calls …
```

## Sample Conversations

### Account inquiry (single intent)

```
Nova:    "Hi, this is Nova from B3Networks. How can I help you today?"
         → look_up_customer(phone_number="+6591001001")
Nova:    "Hi Alex! I can see you're on our Business plan with a balance of $0.
         What can I help you with?"
Caller:  "Just checking my account, that's all."
Nova:    "Great — everything looks good on your account. Is there anything else?"
Caller:  "No, thanks."
Nova:    "Thanks for calling, Alex. Have a great day!"
         → end_call(summary="Account status check — no issues")
```

### Ticket + billing (multi-intent)

```
Nova:    "Hi, this is Nova from B3Networks. How can I help?"
         → look_up_customer(phone_number="+6591001001")
Nova:    "Hi Alex! How can I help you today?"
Caller:  "I've been having WebSocket issues, TKT-1001. Any update?"
         → get_ticket_details(ticket_id="TKT-1001")
Nova:    "I see ticket TKT-1001 — WebSocket disconnection during peak hours.
         It's currently open with high priority. Our engineering team is
         investigating. Anything else I can help with?"
Caller:  "Yeah, when is my next invoice due?"
         → get_billing_summary(phone_number="+6591001001")
Nova:    "Your next invoice, INV-2026-0042 for $299, is due on May 15.
         Your previous invoice is already paid. Anything else?"
Caller:  "That's it."
         → end_call(summary="Checked ticket TKT-1001 status, provided billing info")
```

### FAQ lookup (technical question)

```
Caller:  "How do I handle WebSocket reconnection in the SDK?"
         → search_faq(query="WebSocket reconnection")
Nova:    "The Telcoflow SDK handles reconnection automatically with
         exponential backoff. If you're seeing persistent issues, check your
         internet stability and firewall rules. For high-availability setups,
         Telcoflow Cloud has built-in redundancy. Want me to create a ticket
         if you're still experiencing problems?"
Caller:  "No, that helps. Thanks!"
         → end_call(summary="Answered FAQ about WebSocket reconnection")
```

### Unknown caller (no account)

```
Nova:    "Hi, this is Nova from B3Networks. How can I help?"
         → look_up_customer(phone_number="+6590000000")
Nova:    "I wasn't able to find an account with your number, but I'm still
         happy to help. What can I do for you?"
Caller:  "I want to know about your SDK pricing."
         → search_faq(query="pricing plans")
Nova:    "We have three plans: Starter at $29.90 per month with 1000 minutes,
         Business at $299 per month with 10,000 minutes and priority support,
         and Enterprise at $899 per month with unlimited minutes and a
         dedicated SLA. Would you like more details on any of these?"
```

## Seed Data

### Customers

| Name | Phone | Plan | Balance |
|---|---|---|---|
| Alex Chen | +6591001001 | Business | $0 |
| Maria Santos | +6591002002 | Enterprise | $450 (overdue) |
| James Wong | +6591003003 | Starter | $0 |
| Priya Sharma | +6591004004 | Business | $0 |

### Products

| Name | Category |
|---|---|
| Telcoflow SDK | Developer Tools |
| Telcoflow Connect | Telephony |
| Telcoflow Analytics | Analytics |
| Telcoflow Cloud | Infrastructure |

### Open Tickets

| ID | Customer | Subject | Priority |
|---|---|---|---|
| TKT-1001 | Alex Chen | WebSocket disconnection during peak hours | High |
| TKT-1002 | Maria Santos | Need additional SIP trunks provisioned | Medium |
| TKT-1004 | Priya Sharma | SDK upgrade 0.22→0.24 breaking changes | Medium |

## Configuration

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | *required* | Google API key |
| `WSS_API_KEY` | *required* | Telcoflow API key |
| `WSS_CONNECTOR_UUID` | *required* | Telcoflow connector UUID |
| `GEMINI_MODEL` | `gemini-2.5-flash-native-audio-preview-12-2025` | Gemini Live model |
| `SAMPLE_RATE` | `24000` | Audio sample rate (Hz) |
| `BUSINESS_NAME` | `B3Networks` | Company name in greetings |
| `DB_PATH` | `nova.db` | SQLite database path |
