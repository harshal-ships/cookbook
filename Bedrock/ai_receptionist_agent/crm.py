"""CRM backends for the AI receptionist (JSON demo or Pipedrive)."""
from __future__ import annotations

import json
import os
import re
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_PHONE_COUNTRY_CODE = os.getenv("DEFAULT_PHONE_COUNTRY_CODE", "91").strip()
CUSTOMERS_PATH = Path(os.getenv("CUSTOMERS_PATH", "customers.json")).resolve()
LEADS_PATH = Path(os.getenv("LEADS_PATH", "leads.json")).resolve()


def normalize_phone(value: str | None) -> str:
    if not value:
        return ""
    digits = re.sub(r"\D", "", value)
    if len(digits) == 10 and DEFAULT_PHONE_COUNTRY_CODE:
        digits = DEFAULT_PHONE_COUNTRY_CODE + digits
    return digits


def phone_search_terms(value: str | None) -> list[str]:
    """Build search variants for CRM phone lookup."""
    normalized = normalize_phone(value)
    if not normalized:
        return []
    terms = {normalized}
    if len(normalized) > 10:
        terms.add(normalized[-10:])
    if value:
        raw = value.strip()
        if raw:
            terms.add(raw)
        if not raw.startswith("+") and normalized:
            terms.add(f"+{normalized}")
    return [term for term in terms if len(term) >= 2]


class CRMBackend(ABC):
    @abstractmethod
    async def get_customer_by_phone(self, phone: str | None) -> dict[str, Any] | None:
        """Return unified customer: id, name, phone, company, preferred_department."""

    @abstractmethod
    async def get_open_tickets(self, customer_id: str) -> list[dict[str, Any]]:
        """Return open items: id, subject, status."""

    @abstractmethod
    async def create_lead(self, phone: str | None, call_id: str) -> dict[str, Any]:
        """Create or register an unknown caller."""

    @abstractmethod
    async def update_lead_name(self, phone: str | None, name: str) -> bool:
        """Attach a name to the caller record."""


class JsonCRMBackend(CRMBackend):
    """Local customers.json + leads.json (demo / offline)."""

    def __init__(
        self,
        customers_path: Path = CUSTOMERS_PATH,
        leads_path: Path = LEADS_PATH,
    ):
        self.customers_path = customers_path
        self.leads_path = leads_path

    def _load_customers(self) -> dict[str, Any]:
        return json.loads(self.customers_path.read_text(encoding="utf-8"))

    def _load_leads(self) -> dict[str, Any]:
        if not self.leads_path.exists():
            return {"leads": []}
        return json.loads(self.leads_path.read_text(encoding="utf-8"))

    def _save_leads(self, data: dict[str, Any]) -> None:
        self.leads_path.parent.mkdir(parents=True, exist_ok=True)
        self.leads_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    async def get_customer_by_phone(self, phone: str | None) -> dict[str, Any] | None:
        target = normalize_phone(phone)
        if not target:
            return None
        for customer in self._load_customers().get("customers", []):
            if normalize_phone(customer.get("phone")) == target:
                return customer
        return None

    async def get_open_tickets(self, customer_id: str) -> list[dict[str, Any]]:
        return [
            ticket
            for ticket in self._load_customers().get("tickets", [])
            if ticket.get("customer_id") == customer_id and ticket.get("status") == "open"
        ]

    async def create_lead(self, phone: str | None, call_id: str) -> dict[str, Any]:
        data = self._load_leads()
        lead = {
            "id": str(uuid.uuid4()),
            "phone": phone,
            "phone_normalized": normalize_phone(phone),
            "source": "inbound_call",
            "first_call_id": call_id,
            "created_at": datetime.now().isoformat(),
            "name": None,
        }
        data.setdefault("leads", []).append(lead)
        self._save_leads(data)
        return lead

    async def update_lead_name(self, phone: str | None, name: str) -> bool:
        target = normalize_phone(phone)
        data = self._load_leads()
        updated = False
        for lead in data.get("leads", []):
            if lead.get("phone_normalized") == target and not lead.get("name"):
                lead["name"] = name.strip()
                lead["updated_at"] = datetime.now().isoformat()
                updated = True
        if updated:
            self._save_leads(data)
        return updated


def create_crm_backend() -> CRMBackend:
    backend = os.getenv("CRM_BACKEND", "json").strip().lower()
    if backend == "pipedrive":
        from pipedrive_crm import PipedriveCRMBackend

        return PipedriveCRMBackend()
    return JsonCRMBackend()
