"""
Tool factory for the Nova customer care agent.

`create_nova_tools()` returns a list of async callables that the Google ADK
Agent registers as function-calling tools.  ADK inspects each function's
name, docstring, and type hints to build the schema automatically — no
manual `function_declarations` dict needed.

Each tool is a closure over the shared DB handle, the caller's phone number,
and a `should_end` event used to coordinate call teardown.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from database import NovaDB

logger = logging.getLogger("nova.tools")


@dataclass
class CallMeta:
    """Mutable per-call state shared between tools and the agent loop."""

    caller_phone: str
    customer_id: str = ""
    summary: str = ""
    tools_used: list[str] | None = None

    def __post_init__(self) -> None:
        if self.tools_used is None:
            self.tools_used = []

    def record(self, tool_name: str) -> None:
        self.tools_used.append(tool_name)


def create_nova_tools(
    db: NovaDB,
    meta: CallMeta,
    should_end: asyncio.Event,
) -> list:
    """Build per-call tool closures that ADK will auto-register."""

    async def look_up_customer(phone_number: str) -> dict:
        """Look up a customer account by phone number.
        Returns the customer name, email, plan, balance, and account status.
        Always call this first to identify the caller."""
        meta.record("look_up_customer")
        customer = await db.get_customer_by_phone(phone_number)
        if not customer:
            return {"found": False, "message": "No account found for this number."}
        meta.customer_id = customer.id
        return {
            "found": True,
            "name": customer.name,
            "email": customer.email,
            "plan": customer.plan,
            "balance": f"${customer.balance:.2f}",
            "status": customer.status,
        }

    async def get_ticket_details(ticket_id: str) -> dict:
        """Get the details of a support ticket by its ticket ID (e.g. TKT-1001)."""
        meta.record("get_ticket_details")
        ticket = await db.get_ticket(ticket_id)
        if not ticket:
            return {"found": False, "message": f"No ticket found with ID {ticket_id}."}
        return {
            "found": True,
            "id": ticket.id,
            "subject": ticket.subject,
            "description": ticket.description,
            "status": ticket.status,
            "priority": ticket.priority,
            "created": ticket.created_at,
            "updated": ticket.updated_at,
        }

    async def list_customer_tickets(phone_number: str) -> dict:
        """List all support tickets for a customer, identified by phone number."""
        meta.record("list_customer_tickets")
        customer = await db.get_customer_by_phone(phone_number)
        if not customer:
            return {"found": False, "tickets": []}
        tickets = await db.list_tickets_for_customer(customer.id)
        if not tickets:
            return {"found": True, "tickets": [], "message": f"{customer.name} has no tickets."}
        return {
            "found": True,
            "customer_name": customer.name,
            "tickets": [
                {
                    "id": t.id,
                    "subject": t.subject,
                    "status": t.status,
                    "priority": t.priority,
                }
                for t in tickets
            ],
        }

    async def create_ticket(subject: str, description: str, priority: str) -> dict:
        """Create a new support ticket. Priority must be low, medium, or high.
        Use when the issue cannot be fully resolved during this call."""
        meta.record("create_ticket")
        cust_id = meta.customer_id or ""
        ticket = await db.create_ticket(
            customer_id=cust_id,
            subject=subject,
            description=description,
            priority=priority,
        )
        return {
            "status": "created",
            "ticket_id": ticket.id,
            "message": f"Ticket {ticket.id} created. The team will follow up within 24 hours.",
        }

    async def get_billing_summary(phone_number: str) -> dict:
        """Get billing and invoice information for a customer by phone number."""
        meta.record("get_billing_summary")
        customer = await db.get_customer_by_phone(phone_number)
        if not customer:
            return {"found": False, "message": "No account found."}
        records = await db.get_billing_for_customer(customer.id)
        if not records:
            return {"found": True, "invoices": [], "message": "No billing records found."}
        return {
            "found": True,
            "customer_name": customer.name,
            "plan": customer.plan,
            "invoices": [
                {
                    "invoice": r.invoice_number,
                    "amount": f"${r.amount:.2f}",
                    "due_date": r.due_date,
                    "status": r.status,
                }
                for r in records
            ],
        }

    async def get_product_info(product_name: str) -> dict:
        """Get information about a B3Networks product or service by name.
        Use when the caller asks what a product does or how it works."""
        meta.record("get_product_info")
        product = await db.get_product(product_name)
        if not product:
            products = await db.list_products()
            return {
                "found": False,
                "message": f"No product matching '{product_name}'.",
                "available_products": [p.name for p in products],
            }
        return {
            "found": True,
            "name": product.name,
            "category": product.category,
            "description": product.description,
        }

    async def search_faq(query: str) -> dict:
        """Search the knowledge base for answers to common questions.
        Use when the caller asks a technical or product question."""
        meta.record("search_faq")
        results = await db.search_faq(query)
        if not results:
            return {"results": [], "message": "No matching FAQ entries found."}
        return {
            "results": [
                {"question": f.question, "answer": f.answer, "category": f.category}
                for f in results[:3]
            ]
        }

    async def update_customer_email(phone_number: str, new_email: str) -> dict:
        """Update a customer's email address on file."""
        meta.record("update_customer_email")
        customer = await db.get_customer_by_phone(phone_number)
        if not customer:
            return {"success": False, "message": "Customer not found."}
        ok = await db.update_customer_email(customer.id, new_email)
        if ok:
            return {"success": True, "message": f"Email updated to {new_email}."}
        return {"success": False, "message": "Update failed."}

    async def end_call(summary: str) -> dict:
        """End the call. Call this exactly once as your final action after
        saying goodbye. Provide a brief summary of what was handled."""
        meta.record("end_call")
        meta.summary = summary
        should_end.set()
        return {"status": "ending"}

    return [
        look_up_customer,
        get_ticket_details,
        list_customer_tickets,
        create_ticket,
        get_billing_summary,
        get_product_info,
        search_faq,
        update_customer_email,
        end_call,
    ]
