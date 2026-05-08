"""
Async SQLite database for the Nova customer care agent.

Tables: customers, tickets, products, billing, faq, interactions.
Ships with seed data modelled on a B3Networks customer base.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

import aiosqlite

logger = logging.getLogger("nova.db")

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class Customer:
    id: str
    name: str
    email: str
    phone: str
    plan: str
    balance: float
    status: str
    created_at: str


@dataclass
class Ticket:
    id: str
    customer_id: str
    subject: str
    description: str
    status: str
    priority: str
    created_at: str
    updated_at: str


@dataclass
class Product:
    id: str
    name: str
    category: str
    description: str
    status: str


@dataclass
class BillingRecord:
    id: str
    customer_id: str
    invoice_number: str
    amount: float
    due_date: str
    status: str


@dataclass
class FAQ:
    id: str
    question: str
    answer: str
    category: str


@dataclass
class Interaction:
    id: str
    customer_id: str
    call_id: str
    summary: str
    tools_used: str
    duration_seconds: float
    created_at: str


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    email       TEXT,
    phone       TEXT UNIQUE NOT NULL,
    plan        TEXT NOT NULL,
    balance     REAL DEFAULT 0,
    status      TEXT DEFAULT 'active',
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tickets (
    id          TEXT PRIMARY KEY,
    customer_id TEXT,
    subject     TEXT NOT NULL,
    description TEXT,
    status      TEXT DEFAULT 'open',
    priority    TEXT DEFAULT 'medium',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);

CREATE TABLE IF NOT EXISTS products (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    category    TEXT,
    description TEXT,
    status      TEXT DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS billing (
    id             TEXT PRIMARY KEY,
    customer_id    TEXT NOT NULL,
    invoice_number TEXT NOT NULL,
    amount         REAL NOT NULL,
    due_date       TEXT NOT NULL,
    status         TEXT DEFAULT 'pending',
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);

CREATE TABLE IF NOT EXISTS faq (
    id       TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    answer   TEXT NOT NULL,
    category TEXT
);

CREATE TABLE IF NOT EXISTS interactions (
    id               TEXT PRIMARY KEY,
    customer_id      TEXT,
    call_id          TEXT,
    summary          TEXT,
    tools_used       TEXT,
    duration_seconds REAL,
    created_at       TEXT NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);
"""

# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------

_SEED = """
-- Customers
INSERT INTO customers VALUES
  ('cust-001','Alex Chen','alex.chen@techcorp.sg','+6591001001','Business',0,'active','{now}'),
  ('cust-002','Maria Santos','maria@santos.co','+6591002002','Enterprise',450,'active','{now}'),
  ('cust-003','James Wong','james.w@startup.io','+6591003003','Starter',0,'active','{now}'),
  ('cust-004','Priya Sharma','priya@devhouse.in','+6591004004','Business',0,'active','{now}');

-- Products
INSERT INTO products VALUES
  ('prod-001','Telcoflow SDK','Developer Tools','Python SDK for building production voice AI agents on top of the Telcoflow platform. Includes WebSocket streaming, call control, event handling, and audio processing.','active'),
  ('prod-002','Telcoflow Connect','Telephony','SIP trunking and phone number management service. Provision numbers, configure routing rules, and connect to carrier networks.','active'),
  ('prod-003','Telcoflow Analytics','Analytics','Real-time call analytics dashboard. Track call volume, intent distribution, resolution rates, and agent performance.','active'),
  ('prod-004','Telcoflow Cloud','Infrastructure','Managed hosting for voice AI workloads. Auto-scaling, low-latency audio processing, and built-in redundancy.','active');

-- Tickets
INSERT INTO tickets VALUES
  ('TKT-1001','cust-001','WebSocket disconnection during peak hours','Agent loses connection to Telcoflow when call volume exceeds 50 concurrent sessions. Reconnect logic triggers but adds 2-3 second gap.','open','high','{now}','{now}'),
  ('TKT-1002','cust-002','Need additional SIP trunks provisioned','Current allocation of 20 trunks is insufficient for holiday season. Requesting increase to 50.','in_progress','medium','{now}','{now}'),
  ('TKT-1003','cust-003','Audio quality issues on international calls','Callers from EU region report choppy audio and occasional one-way audio. Codec negotiation may be failing.','resolved','high','{t_minus_5}','{t_minus_1}'),
  ('TKT-1004','cust-004','SDK upgrade from 0.22 to 0.24 breaking changes','After upgrading, the CALL_TERMINATED event payload changed shape. Need migration guidance.','open','medium','{now}','{now}');

-- Billing
INSERT INTO billing VALUES
  ('bill-001','cust-001','INV-2026-0041',299.00,'{due_current}','paid'),
  ('bill-002','cust-001','INV-2026-0042',299.00,'{due_next}','pending'),
  ('bill-003','cust-002','INV-2026-0038',899.00,'{due_past}','overdue'),
  ('bill-004','cust-002','INV-2026-0039',899.00,'{due_current}','pending'),
  ('bill-005','cust-003','INV-2026-0045',29.90,'{due_current}','paid'),
  ('bill-006','cust-004','INV-2026-0044',299.00,'{due_current}','paid');

-- FAQ
INSERT INTO faq VALUES
  ('faq-001','How do I get started with the Telcoflow SDK?','Install with pip: pip install telcoflow-sdk. Then get your API key from the B3Networks dashboard, create a connector, and follow the Quick Start guide at docs.telcoflow.com.','getting_started'),
  ('faq-002','How do I configure SIP trunking?','Log into the Telcoflow Connect dashboard. Navigate to Trunking, click Add Trunk, enter your endpoint IP and port, select a codec profile, and save. The trunk will be provisioned within 60 seconds.','telephony'),
  ('faq-003','What are the available pricing plans?','Starter plan at $29.90 per month includes 1000 minutes and basic analytics. Business plan at $299 per month includes 10000 minutes, priority support, and full analytics. Enterprise plan at $899 per month includes unlimited minutes, dedicated support, SLA guarantees, and custom integrations.','billing'),
  ('faq-004','How do I troubleshoot audio quality issues?','Check your network bandwidth (minimum 100 kbps per concurrent call). Verify the sample rate matches between your agent and the Telcoflow config (default 24000 Hz). Ensure your firewall allows WebSocket connections on port 443. If using international routes, check codec negotiation logs in the Analytics dashboard.','troubleshooting'),
  ('faq-005','How do I handle WebSocket reconnection?','The Telcoflow SDK handles reconnection automatically with exponential backoff. If you experience persistent disconnections, check your internet stability, verify your API key has not expired, and ensure your firewall allows persistent WebSocket connections. For high-availability setups, use Telcoflow Cloud.','troubleshooting'),
  ('faq-006','How do I add tools and function calling to my voice agent?','Define your tools as function declarations in the Gemini Live config or as Python functions with the Google ADK. The AI model will call your tools during the conversation. See the Use Cases section at docs.telcoflow.com for production examples.','developer'),
  ('faq-007','What is the difference between connect() and close()?','connect() bridges the caller to the original callee (the number they dialed). close() removes your agent from the call while keeping the caller and callee connected. Use them together for warm transfers: connect() first, then close().','developer'),
  ('faq-008','How do I monitor call quality in production?','Use Telcoflow Analytics to track MOS scores, packet loss, jitter, and latency per call. Set up alerts for quality degradation. For real-time monitoring, subscribe to CALL_QUALITY events in your agent code.','operations');
""".strip()

# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------


class NovaDB:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(_SCHEMA)

        async with self._conn.execute("SELECT count(*) AS cnt FROM customers") as cur:
            row = await cur.fetchone()
            if row["cnt"] == 0:
                now = datetime.utcnow()
                seed = _SEED.format(
                    now=now.strftime("%Y-%m-%d %H:%M:%S"),
                    t_minus_5=(now - timedelta(days=5)).strftime("%Y-%m-%d %H:%M:%S"),
                    t_minus_1=(now - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S"),
                    due_past=(now - timedelta(days=10)).strftime("%Y-%m-%d"),
                    due_current=now.strftime("%Y-%m-%d"),
                    due_next=(now + timedelta(days=30)).strftime("%Y-%m-%d"),
                )
                await self._conn.executescript(seed)

        await self._conn.commit()
        logger.info("NovaDB ready at %s", self._db_path)

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            logger.info("NovaDB closed")

    # -- customers ----------------------------------------------------------

    async def get_customer_by_phone(self, phone: str) -> Customer | None:
        async with self._conn.execute(
            "SELECT * FROM customers WHERE phone = ?", (phone,)
        ) as cur:
            row = await cur.fetchone()
            return Customer(**row) if row else None

    # -- tickets ------------------------------------------------------------

    async def get_ticket(self, ticket_id: str) -> Ticket | None:
        async with self._conn.execute(
            "SELECT * FROM tickets WHERE id = ?", (ticket_id,)
        ) as cur:
            row = await cur.fetchone()
            return Ticket(**row) if row else None

    async def list_tickets_for_customer(self, customer_id: str) -> list[Ticket]:
        async with self._conn.execute(
            "SELECT * FROM tickets WHERE customer_id = ? ORDER BY created_at DESC",
            (customer_id,),
        ) as cur:
            return [Ticket(**r) for r in await cur.fetchall()]

    async def create_ticket(
        self,
        customer_id: str,
        subject: str,
        description: str,
        priority: str = "medium",
    ) -> Ticket:
        tid = f"TKT-{uuid.uuid4().hex[:6].upper()}"
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        await self._conn.execute(
            "INSERT INTO tickets VALUES (?,?,?,?,?,?,?,?)",
            (tid, customer_id, subject, description, "open", priority, now, now),
        )
        await self._conn.commit()
        logger.info("Ticket %s created for %s", tid, customer_id)
        return Ticket(tid, customer_id, subject, description, "open", priority, now, now)

    # -- products -----------------------------------------------------------

    async def get_product(self, name: str) -> Product | None:
        async with self._conn.execute(
            "SELECT * FROM products WHERE LOWER(name) LIKE ?",
            (f"%{name.lower()}%",),
        ) as cur:
            row = await cur.fetchone()
            return Product(**row) if row else None

    async def list_products(self) -> list[Product]:
        async with self._conn.execute(
            "SELECT * FROM products WHERE status = 'active'"
        ) as cur:
            return [Product(**r) for r in await cur.fetchall()]

    # -- billing ------------------------------------------------------------

    async def get_billing_for_customer(self, customer_id: str) -> list[BillingRecord]:
        async with self._conn.execute(
            "SELECT * FROM billing WHERE customer_id = ? ORDER BY due_date DESC",
            (customer_id,),
        ) as cur:
            return [BillingRecord(**r) for r in await cur.fetchall()]

    # -- faq ----------------------------------------------------------------

    async def search_faq(self, query: str) -> list[FAQ]:
        words = query.lower().split()
        if not words:
            return []
        conditions = " OR ".join(
            ["LOWER(question || ' ' || answer || ' ' || category) LIKE ?"] * len(words)
        )
        params = [f"%{w}%" for w in words]
        async with self._conn.execute(
            f"SELECT * FROM faq WHERE {conditions}", params
        ) as cur:
            return [FAQ(**r) for r in await cur.fetchall()]

    # -- interactions -------------------------------------------------------

    async def log_interaction(
        self,
        call_id: str,
        customer_id: str,
        summary: str,
        tools_used: str,
        duration_seconds: float,
    ) -> None:
        iid = f"INT-{uuid.uuid4().hex[:8]}"
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        await self._conn.execute(
            "INSERT INTO interactions VALUES (?,?,?,?,?,?,?)",
            (iid, customer_id, call_id, summary, tools_used, duration_seconds, now),
        )
        await self._conn.commit()
        logger.info("Interaction %s logged", iid)

    # -- customer updates ---------------------------------------------------

    async def update_customer_email(self, customer_id: str, new_email: str) -> bool:
        result = await self._conn.execute(
            "UPDATE customers SET email = ? WHERE id = ?", (new_email, customer_id)
        )
        await self._conn.commit()
        return result.rowcount > 0
