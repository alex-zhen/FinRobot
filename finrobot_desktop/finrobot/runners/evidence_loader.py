"""Load sourced evidence files into official amount rows.

Precomputed TTM totals in those files are not inputs. Collection flags such as
``usable_as_finrobot_input`` are ignored. Only the statutory cash-flow capex
line can be bridged, and a full year alone is not a trailing twelve months.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from finrobot.runners.official_inputs import OfficialAmount

_CURRENCY = {"RMB": "CNY", "CNY": "CNY", "USD": "USD"}


def _iso_currency(raw: str) -> str:
    return _CURRENCY.get(raw.strip().upper(), raw.strip().upper())


def _url_from_location(location: dict[str, Any] | None) -> str | None:
    if not location:
        return None
    for key in ("sec_html_url", "url", "url_hkex", "url_ir"):
        value = location.get(key)
        if isinstance(value, str) and value.startswith(("https://", "http://")):
            return value
    return None


def load_tencent_statutory_capex(path: Path) -> list[OfficialAmount]:
    """PPE/CIP/investment-property cash outflow only. Management capex is skipped."""
    document = json.loads(path.read_text())
    label = document["line_item_definitions"]["capex_cash_ppe_cip_ip"]["label"]
    sources = document["sources"]
    rows: list[OfficialAmount] = []
    for period_key, period_type in (("FY2025", "fy"), ("H1_2025", "ytd"), ("H1_2026", "ytd")):
        period = document["periods"][period_key]
        item = period["items"]["capex_cash_ppe_cip_ip"]
        source = sources[period["source"]]
        url = source.get("url") or source.get("url_hkex") or source.get("url_ir")
        magnitude = float(item["amount_rmb_million"])
        signed = -magnitude if item.get("cash_flow_sign") == "outflow" else magnitude
        rows.append(
            OfficialAmount(
                ticker="0700.HK",
                source_url=url,
                source_kind=str(source["title"]),
                statement_line=label,
                line_item="capital_expenditure",
                line_role="statutory_capex",
                period_start=date.fromisoformat(period["period_start"]),
                period_end=date.fromisoformat(period["period_end"]),
                period_type=period_type,  # type: ignore[arg-type]
                currency="CNY",
                unit="million",
                value=signed,
            )
        )
    return rows


def load_finrobot_evidence_capex(path: Path) -> list[OfficialAmount]:
    """Map a finrobot_input_evidence file. The usable flag is not consulted."""
    document = json.loads(path.read_text())
    rows: list[OfficialAmount] = []
    for issuer in document["issuers"].values():
        ticker = str(issuer["tickers"][0])
        for period in issuer["periods"].values():
            for field in period["fields"]:
                role = _role_for_field(str(field["field_id"]), str(field.get("line_item") or ""))
                if role is None:
                    continue
                url = _url_from_location(field.get("source_location"))
                if url is None:
                    continue
                raw_value = field.get("value")
                if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                    continue
                unit = str(field.get("unit") or "")
                if unit not in {"ones", "thousand", "million", "billion"}:
                    continue
                period_type = _period_type(str(field["period_type"]))
                start = field.get("period_start") or field["period_end"]
                if not start or not field.get("period_end"):
                    continue
                rows.append(
                    OfficialAmount(
                        ticker=ticker,
                        source_url=url,
                        source_kind=str(field.get("official_source") or "filing"),
                        statement_line=str(field["line_item"]),
                        line_item="capital_expenditure",
                        line_role=role,  # type: ignore[arg-type]
                        period_start=date.fromisoformat(str(start)),
                        period_end=date.fromisoformat(str(field["period_end"])),
                        period_type=period_type,  # type: ignore[arg-type]
                        currency=_iso_currency(str(field["currency"])),
                        unit=unit,  # type: ignore[arg-type]
                        value=float(raw_value),
                    )
                )
    return rows


def load_tencent_period_end_shares(path: Path) -> tuple[float, date] | None:
    """Issued ordinary shares at the interim period end.

    This is not a weighted-average share count and it is not FY + YTD − prior YTD.
    """
    document = json.loads(path.read_text(encoding="utf-8"))
    block = document["periods"]["H1_2026"]["shares"]["issued_and_fully_paid_period_end"]
    return float(block["shares"]), date.fromisoformat(str(block["as_of"]))


def _period_type(raw: str) -> str:
    text = raw.strip().lower()
    if text == "fy":
        return "fy"
    if text in {"ytd_six_months", "half", "h1"}:
        return "ytd"
    return "quarter"


def _role_for_field(field_id: str, line_item: str) -> str | None:
    token = f"{field_id} {line_item}".lower()
    if "self_calculated" in token or "self-calculated" in token:
        return "self_calculated_fcf"
    if "company_defined_fcf" in token or "free cash flow" in token:
        return "company_fcf"
    if "management_capex" in token or "capital expenditures (d)" in token:
        return "management_capex"
    if "narrow_ppe" in token or "excluding land use rights" in token:
        return "fcf_recon_ppe"
    if "capex_acquisitions_of_land" in token or "purchases_related_to_ppe" in token:
        return "statutory_capex"
    if "capex_cash_ppe" in token:
        return "statutory_capex"
    return None
