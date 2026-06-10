"""Pipedrive CRM backend for AI receptionist.

API docs: https://developers.pipedrive.com/docs/api/v1/Persons
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any
from urllib import error, parse, request

from crm import CRMBackend, normalize_phone, phone_search_terms

logger = logging.getLogger(__name__)


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


class PipedriveCRMBackend(CRMBackend):
    """Pipedrive persons, deals, and leads for inbound call lookup."""

    def __init__(self) -> None:
        self.api_token = require_env("PIPEDRIVE_API_TOKEN")
        domain = require_env("PIPEDRIVE_COMPANY_DOMAIN").removesuffix(".pipedrive.com")
        self.base_url = f"https://{domain}.pipedrive.com/api/v1"
        self.lead_title_prefix = os.getenv("PIPEDRIVE_LEAD_TITLE_PREFIX", "Inbound call")
        self._person_cache: dict[str, int] = {}

    def _api_request(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        params = dict(query or {})
        params["api_token"] = self.api_token
        url = f"{self.base_url}{path}?{parse.urlencode(params)}"
        data = None
        headers = {"Content-Type": "application/json"}
        if body is not None:
            data = json.dumps(body).encode("utf-8")

        req = request.Request(url, data=data, headers=headers, method=method.upper())
        try:
            with request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Pipedrive API error {exc.code} on {path}: {detail}") from exc

        if not payload.get("success", True) and payload.get("error"):
            raise RuntimeError(f"Pipedrive API returned error on {path}: {payload.get('error')}")
        return payload

    async def _request(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._api_request, method, path, query=query, body=body
        )

    def _map_person(self, person: dict[str, Any]) -> dict[str, Any]:
        org_name = None
        org_id = person.get("org_id")
        if isinstance(person.get("organization"), dict):
            org_name = person["organization"].get("name")
        phones = person.get("phone") or []
        primary_phone = phones[0].get("value") if phones else None
        person_id = person.get("id")
        if person_id is not None:
            self._person_cache[normalize_phone(primary_phone)] = int(person_id)
        return {
            "id": str(person_id),
            "name": person.get("name") or "Unknown",
            "phone": primary_phone,
            "company": org_name or (f"org-{org_id}" if org_id else "n/a"),
            "preferred_department": "general",
            "pipedrive_person_id": person_id,
            "pipedrive_org_id": org_id,
        }

    async def _search_person(self, phone: str | None) -> dict[str, Any] | None:
        for term in phone_search_terms(phone):
            payload = await self._request(
                "GET",
                "/persons/search",
                query={"term": term, "fields": "phone", "limit": 5},
            )
            items = payload.get("data", {}).get("items") or []
            target = normalize_phone(phone)
            for item in items:
                person = item.get("item") or item
                if person.get("type") and person.get("type") != "person":
                    continue
                for phone_entry in person.get("phone") or []:
                    if normalize_phone(phone_entry.get("value")) == target:
                        return person
            if items:
                first = items[0].get("item") or items[0]
                if first.get("id"):
                    return first
        return None

    async def get_customer_by_phone(self, phone: str | None) -> dict[str, Any] | None:
        person = await self._search_person(phone)
        if not person:
            return None
        person_id = person.get("id")
        if person_id:
            detail = await self._request("GET", f"/persons/{person_id}")
            if detail.get("data"):
                person = detail["data"]
        return self._map_person(person)

    async def get_open_tickets(self, customer_id: str) -> list[dict[str, Any]]:
        person_id = customer_id.removeprefix("pipedrive:") if customer_id else ""
        if not person_id.isdigit():
            person_id = customer_id
        payload = await self._request(
            "GET",
            f"/persons/{person_id}/deals",
            query={"status": "open", "limit": 20},
        )
        deals = payload.get("data") or []
        return [
            {
                "id": f"DEAL-{deal.get('id')}",
                "subject": deal.get("title") or "Open deal",
                "status": deal.get("status") or "open",
                "value": deal.get("value"),
                "currency": deal.get("currency"),
            }
            for deal in deals
        ]

    async def _create_person(self, phone: str | None, name: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if phone:
            body["phone"] = [{"value": phone, "primary": True, "label": "mobile"}]
        if name:
            body["name"] = name
        else:
            body["name"] = phone or "Inbound caller"
        payload = await self._request("POST", "/persons", body=body)
        person = payload.get("data") or {}
        return self._map_person(person)

    async def _create_pipedrive_lead(self, person_id: int, call_id: str) -> dict[str, Any] | None:
        title = f"{self.lead_title_prefix} {call_id}"
        try:
            payload = await self._request(
                "POST",
                "/leads",
                body={
                    "title": title,
                    "person_id": person_id,
                    "origin": "API",
                    "origin_id": call_id,
                },
            )
            return payload.get("data")
        except RuntimeError:
            logger.warning("Pipedrive leads API unavailable; person created without lead")
            return None

    async def create_lead(self, phone: str | None, call_id: str) -> dict[str, Any]:
        existing = await self._search_person(phone)
        if existing:
            mapped = self._map_person(existing)
            return {
                "id": mapped["id"],
                "phone": phone,
                "phone_normalized": normalize_phone(phone),
                "source": "pipedrive_existing_person",
                "first_call_id": call_id,
                "pipedrive_person_id": mapped.get("pipedrive_person_id"),
                "created": False,
            }

        person = await self._create_person(phone)
        person_id = person.get("pipedrive_person_id")
        lead = None
        if person_id:
            lead = await self._create_pipedrive_lead(int(person_id), call_id)

        return {
            "id": person["id"],
            "phone": phone,
            "phone_normalized": normalize_phone(phone),
            "source": "pipedrive",
            "first_call_id": call_id,
            "pipedrive_person_id": person_id,
            "pipedrive_lead_id": lead.get("id") if lead else None,
            "created": True,
        }

    async def update_lead_name(self, phone: str | None, name: str) -> bool:
        clean_name = name.strip()
        if not clean_name:
            return False

        person_id = self._person_cache.get(normalize_phone(phone))
        if not person_id:
            person = await self._search_person(phone)
            person_id = person.get("id") if person else None
        if not person_id:
            return False

        await self._request(
            "PUT",
            f"/persons/{person_id}",
            body={"name": clean_name},
        )
        return True
