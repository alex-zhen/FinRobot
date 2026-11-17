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

from finrobot.runners.official_inputs import (
    OfficialAmount,
    bridge_same_caption,
    fiscal_period_end_share_use,
    later_quote_share_estimate,
    period_end_ordinary_share_block,
    weekend_month_end_share_bridge,
    self_calculated_price_times_shares,
)

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


def tencent_official_flow_facts(path: Path) -> dict[str, Any]:
    """Recompute Tencent flow TTM from FY2025 + H1 2026 − H1 2025.

    Capex is the PPE/CIP/investment-property cash line only. The precomputed
    TTM block in the file is not the source of these numbers.
    """
    document = json.loads(path.read_text(encoding="utf-8"))
    periods = document["periods"]

    def _amount(period_key: str, item_key: str) -> float:
        return float(periods[period_key]["items"][item_key]["amount_rmb_million"])

    flows: dict[str, float] = {}
    for key in ("revenue", "operating_cash_flow", "capex_cash_ppe_cip_ip"):
        flows[key] = _amount("FY2025", key) + _amount("H1_2026", key) - _amount("H1_2025", key)
    return {
        "ticker": "0700.HK",
        "currency": "CNY",
        "unit": "million",
        "window": "2025-07-01 to 2026-06-30",
        "construction": "fy_ytd_bridge",
        "revenue": flows["revenue"],
        "operating_cash_flow": flows["operating_cash_flow"],
        "statutory_capex_outflow": flows["capex_cash_ppe_cip_ip"],
        "capex_line": (
            "Purchase of/prepayments for property, plant and equipment, "
            "construction in progress and investment properties"
        ),
        "capex_present": True,
    }


def segment_business_background(path: Path) -> dict[str, Any]:
    """Segment names and limits only. Not a peer list and not a valuation."""
    document = json.loads(path.read_text(encoding="utf-8"))
    issuers = []
    for issuer in document.get("issuers", []):
        segments = issuer.get("official_reportable_segments", {}).get("segments", [])
        issuers.append(
            {
                "name": issuer.get("common_name"),
                "tickers": [item.get("symbol") for item in issuer.get("tickers", [])],
                "segment_names": [item.get("name") for item in segments if item.get("name")],
            }
        )
    return {
        "officially_named_listed_peers": False,
        "not_for_multiple_average": True,
        "subsidiaries_are_not_external_peers": True,
        "unlisted_names_are_not_listed_multiples": True,
        "semiconductor_roles_apply": False,
        "issuers": issuers,
        "use": "business background for a non-valuation note only",
    }


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


def _optional_bridge(
    *,
    role: str,
    caption: str,
    fy_end: date,
    fy_value: float | None,
    current_end: date,
    current_value: float | None,
    prior_end: date,
    prior_value: float | None,
    source: str,
    as_capex_outflow: bool,
) -> dict[str, Any]:
    if fy_value is None or current_value is None or prior_value is None:
        return {
            "role": role,
            "signed_value": None,
            "opens_publication_gate": False,
            "reason": "a component of this caption is missing",
        }
    built = bridge_same_caption(
        caption=caption,
        role=role,
        fy_end=fy_end,
        fy_value=fy_value,
        current_end=current_end,
        current_value=current_value,
        prior_end=prior_end,
        prior_value=prior_value,
        currency="CNY",
        source=source,
        as_capex_outflow=as_capex_outflow,
    )
    built["unit"] = "million"
    return built


def _number(field: dict[str, Any] | None) -> float | None:
    if not field:
        return None
    raw = field.get("value")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return float(raw)


def baba_line_records(path: Path) -> dict[str, Any]:
    """Recompute BABA bridges from component rows. Precomputed TTM totals are not read.

    Statutory capex stays blocked when either quarterly statutory line is absent.
    A same-caption OCF bridge and a narrower PPE bridge are recorded beside it
    and do not open the publication gate. The completed fiscal year is a
    historical baseline, not the latest TTM.
    """
    document = json.loads(path.read_text(encoding="utf-8"))
    periods = document["periods"]
    fy = periods["FY2026"]["fields"]
    current = periods["FY2027Q1"]["fields"]
    prior = periods["FY2026Q1"]["fields"]
    source = str(document["documents_searched"]["fy2026_20f"]["sec_html_url"])
    fy_end = date.fromisoformat(str(periods["FY2026"]["period_end"]))
    current_end = date.fromisoformat(str(periods["FY2027Q1"]["period_end"]))
    prior_end = date.fromisoformat(str(periods["FY2026Q1"]["period_end"]))
    statutory_fy = _number(fy.get("statutory_capex"))
    statutory_current = _number(current.get("statutory_capex"))
    statutory_prior = _number(prior.get("statutory_capex"))
    statutory_ttm: dict[str, Any] | None = None
    if statutory_fy is not None and statutory_current is not None and statutory_prior is not None:
        statutory_ttm = bridge_same_caption(
            caption="statutory_capex",
            role="statutory_capex",
            fy_end=fy_end,
            fy_value=statutory_fy,
            current_end=current_end,
            current_value=statutory_current,
            prior_end=prior_end,
            prior_value=statutory_prior,
            currency="CNY",
            source=source,
            as_capex_outflow=True,
        )
    narrow = _optional_bridge(
        role="fcf_recon_ppe",
        caption="narrow_fcf_ppe",
        fy_end=fy_end,
        fy_value=_number(fy.get("narrow_fcf_ppe")),
        current_end=current_end,
        current_value=_number(current.get("narrow_fcf_ppe")),
        prior_end=prior_end,
        prior_value=_number(prior.get("narrow_fcf_ppe")),
        source=source,
        as_capex_outflow=True,
    )
    narrow["opens_publication_gate"] = False
    ocf = _optional_bridge(
        role="operating_cash_flow",
        caption="operating_cash_flow",
        fy_end=fy_end,
        fy_value=_number(fy.get("operating_cash_flow")),
        current_end=current_end,
        current_value=_number(current.get("operating_cash_flow")),
        prior_end=prior_end,
        prior_value=_number(prior.get("operating_cash_flow")),
        source=source,
        as_capex_outflow=False,
    )
    return {
        "ticker": "BABA",
        "statutory_ttm_capex": statutory_ttm,
        "statutory_ttm_blocked": statutory_ttm is None
        or not statutory_ttm["opens_publication_gate"],
        "narrow_ppe_alternative": narrow,
        "ocf_ttm": ocf,
        "management_capex_ttm": None,
        "management_mix_refused": (
            "Quarterly MD&A capital expenditures are not the statutory cash-flow "
            "line and are not combined with the full-year statutory amount."
        ),
        "historical_fy_baseline": {
            "kind": "historical_fy_baseline",
            "period_end": periods["FY2026"]["period_end"],
            "statutory_capex_rmb_million": _number(fy.get("statutory_capex")),
            "ocf_rmb_million": _number(fy.get("operating_cash_flow")),
            "usable_as_latest_ttm": False,
            "usable_as_current_analysis": False,
        },
    }


def tencent_same_day_share_record(path: Path, *, valuation_as_of: date) -> dict[str, Any]:
    """Issued shares × the same-day supplementary close. Not an official market cap.

    The figure is historical when ``valuation_as_of`` is a later session. It is
    not written onto the live snapshot and does not set ``share_class_scope``.
    """
    document = json.loads(path.read_text(encoding="utf-8"))
    shares = next(
        item for item in document["items"] if item["item"] == "period_end_issued_ordinary_shares"
    )
    close = next(
        item
        for item in document["items"]
        if item["item"] == "closing_price_supplement_akshare_sina"
    )
    price = float(close["close_conventional_tick"])
    share_count = float(shares["value"])
    as_of = date.fromisoformat(str(shares["as_of_date"]))
    built = self_calculated_price_times_shares(
        shares=share_count,
        share_as_of=as_of,
        price=price,
        price_as_of=date.fromisoformat(str(close["trade_date"])),
        valuation_as_of=valuation_as_of,
        price_is_official_exchange_file=False,
    )
    built["shares"] = share_count
    built["price"] = price
    built["share_class_scope_for_share_date"] = "single_listing"
    built["third_party_market_cap_used"] = False
    value = built["value"]
    built["historical_share_gate"] = period_end_ordinary_share_block(
        shares=share_count,
        share_as_of=as_of,
        price=price,
        price_as_of=date.fromisoformat(str(close["trade_date"])),
        market_cap=float(value) if isinstance(value, (int, float)) else None,
        market_cap_basis="self_calculated_price_times_issued_shares",
    )
    return {"ticker": "0700.HK", **built}


def tencent_october_share_record(path: Path, *, valuation_as_of: date) -> dict[str, Any]:
    """October month-end shares and the prior Friday official close.

    The close is the CLOSING column, not the ask. Pending-cancellation shares
    stay inside the issued total. June award-scheme holdings are not carried
    forward. A later close is not paired with this share count.
    """
    document = json.loads(path.read_text(encoding="utf-8"))
    shares = next(
        item for item in document["items"] if item["item"] == "period_end_issued_ordinary_shares"
    )
    pending = next(
        item
        for item in document["items"]
        if item["item"] == "shares_repurchased_for_cancellation_but_not_yet_cancelled"
    )
    official_close = next(
        item
        for item in document["items"]
        if item["item"] == "official_sehk_closing_price_primary_hkex_file"
    )
    later = next(
        item
        for item in document["items"]
        if item["item"] == "unused_later_prices_not_to_be_paired_with_2026_10_31_shares"
    )
    price = float(official_close["closing"])
    ask = float(official_close["ask"])
    share_count = float(shares["value"])
    share_as_of = date.fromisoformat(str(shares["as_of_date"]))
    price_as_of = date.fromisoformat(str(official_close["trade_date"]))
    built = weekend_month_end_share_bridge(
        shares=share_count,
        share_as_of=share_as_of,
        price=price,
        price_as_of=price_as_of,
        valuation_as_of=valuation_as_of,
        pending_cancellation_included=float(pending["value"]),
    )
    built["ask_not_used"] = ask
    built["award_scheme_shares_carried_forward"] = False
    built["rejected_later_close"] = {
        "trade_date": later["akshare_sina_latest_daily_in_window"]["trade_date"],
        "close": later["akshare_sina_latest_daily_in_window"]["close_conventional_tick"],
        "paired_with_share_as_of": False,
    }
    built["verified_same_day_block"] = period_end_ordinary_share_block(
        shares=share_count,
        share_as_of=share_as_of,
        price=price,
        price_as_of=price_as_of,
        market_cap=None,
    )
    later_bar = later["akshare_sina_latest_daily_in_window"]
    built["later_quote_estimate"] = later_quote_share_estimate(
        shares=share_count,
        share_as_of=share_as_of,
        price=float(later_bar["close_conventional_tick"]),
        price_as_of=date.fromisoformat(str(later_bar["trade_date"])),
    )
    return {"ticker": "0700.HK", **built}


def load_nvda_official_capex(path: Path) -> list[OfficialAmount]:
    """Statutory purchases from FY, current H1, and the 10-Q prior-H1 column.

    Magnitudes marked as cash outflows are stored as signed negatives. Financing
    principal payments are not this line.
    """
    document = json.loads(path.read_text(encoding="utf-8"))
    rows: list[OfficialAmount] = []
    for key, period_type in (
        ("FY2026", "fy"),
        ("FY2027H1_current", "ytd"),
        ("FY2026H1_prior_comparative", "ytd"),
    ):
        period = document["periods"][key]
        field = period["fields"][
            "purchases_related_to_property_and_equipment_and_intangible_assets"
        ]
        magnitude = float(field["value"])
        signed = -magnitude if field.get("sign") == "cash_outflow" else magnitude
        rows.append(
            OfficialAmount(
                ticker="NVDA",
                source_url=str(period["source_url"]),
                source_kind=str(period["source"])[:80],
                statement_line=(
                    "Purchases related to property and equipment and intangible assets"
                ),
                line_item="capital_expenditure",
                line_role="statutory_capex",
                period_start=date.fromisoformat(str(period["period_start"])),
                period_end=date.fromisoformat(str(period["period_end"])),
                period_type=period_type,  # type: ignore[arg-type]
                currency="USD",
                unit="million",
                value=signed,
            )
        )
    return rows


def nvda_official_record(path: Path) -> dict[str, Any]:
    """Recompute official TTM flows from the three columns. The summary block is not read."""
    document = json.loads(path.read_text(encoding="utf-8"))
    periods = document["periods"]
    fy = periods["FY2026"]
    current = periods["FY2027H1_current"]
    prior = periods["FY2026H1_prior_comparative"]
    source = str(fy["source_url"])
    flows: dict[str, Any] = {}
    for caption, role, outflow in (
        ("revenue", "revenue", False),
        ("net_income", "net_income", False),
        ("operating_cash_flow", "operating_cash_flow", False),
        (
            "purchases_related_to_property_and_equipment_and_intangible_assets",
            "statutory_capex",
            True,
        ),
    ):
        fy_field = fy["fields"][caption]
        current_field = current["fields"][caption]
        prior_field = prior["fields"][caption]
        flows[caption] = _signed_bridge(
            caption=caption,
            role=role,
            fy_end=date.fromisoformat(str(fy["period_end"])),
            fy_value=_signed_flow(fy_field),
            current_end=date.fromisoformat(str(current["period_end"])),
            current_value=_signed_flow(current_field),
            prior_end=date.fromisoformat(str(prior["period_end"])),
            prior_value=_signed_flow(prior_field),
            source=source,
            currency="USD",
            as_capex_outflow=outflow,
        )
    shares = current["fields"]["common_stock_outstanding_million"]
    share_as_of = date.fromisoformat(str(shares["as_of"]))
    share_use = fiscal_period_end_share_use(
        shares=float(shares["value"]) * 1_000_000,
        share_as_of=share_as_of,
        price_as_of=None,
    )
    return {
        "ticker": "NVDA",
        "ttm_period_start": "2025-07-28",
        "ttm_period_end": "2026-07-26",
        "currency": "USD",
        "unit": "million",
        "flows": flows,
        "company_labeled_fcf_required": False,
        "independent_market_cap_required": False,
        "official_quarter_ends": ["2025-10-26", "2026-01-25", "2026-04-26", "2026-07-26"],
        "period_end_shares": share_use,
        "financing_principal_payments_not_in_capex": True,
    }


def baba_alternative_records(path: Path) -> dict[str, Any]:
    """Company FCF and narrow PPE TTM. Neither opens the statutory FCFF gate."""
    document = json.loads(path.read_text(encoding="utf-8"))
    fcf_periods = document["company_defined_fcf"]["periods"]
    ppe_periods = document["narrow_ppe"]["periods"]
    source = "https://www.sec.gov/Archives/edgar/data/1577552/000119312526231755/baba-20260331.htm"
    company_fcf = _signed_bridge(
        caption="company_defined_fcf",
        role="company_fcf",
        fy_end=date.fromisoformat(str(fcf_periods["FY2026"]["period_end"])),
        fy_value=float(fcf_periods["FY2026"]["value"]),
        current_end=date.fromisoformat(str(fcf_periods["FY2027Q1"]["period_end"])),
        current_value=float(fcf_periods["FY2027Q1"]["value"]),
        prior_end=date.fromisoformat(str(fcf_periods["FY2026Q1"]["period_end"])),
        prior_value=float(fcf_periods["FY2026Q1"]["value"]),
        source=source,
        currency="CNY",
        as_capex_outflow=False,
    )
    company_fcf["opens_publication_gate"] = False
    company_fcf["usable_as_fcff"] = False
    narrow = _signed_bridge(
        caption="narrow_fcf_ppe",
        role="fcf_recon_ppe",
        fy_end=date.fromisoformat(str(ppe_periods["FY2026"]["period_end"])),
        fy_value=float(ppe_periods["FY2026"]["value"]),
        current_end=date.fromisoformat(str(ppe_periods["FY2027Q1"]["period_end"])),
        current_value=float(ppe_periods["FY2027Q1"]["value"]),
        prior_end=date.fromisoformat(str(ppe_periods["FY2026Q1"]["period_end"])),
        prior_value=float(ppe_periods["FY2026Q1"]["value"]),
        source=source,
        currency="CNY",
        as_capex_outflow=True,
    )
    narrow["opens_publication_gate"] = False
    quarterly_intangibles = fcf_periods["FY2027Q1"]["recon_items_rmb_million"][
        "purchase_of_intangible_assets_ex_acquisitions"
    ]
    return {
        "ticker": "BABA",
        "statutory_ttm_blocked": True,
        "company_defined_fcf_ttm": company_fcf,
        "narrow_ppe_alternative": narrow,
        "management_capex_ttm": None,
        "management_capex_side_note_only": True,
        "quarterly_intangibles_officially_zero": quarterly_intangibles == 0,
        "research_risk": (
            "Company-defined free cash flow for 2025-07-01 to 2026-06-30 is negative. "
            "That is a non-GAAP liquidity observation. It is not FCFF and it does not "
            "open the statutory capex gate."
        ),
        "hard_gaps": [
            "statutory cash-flow capex TTM is unpresented on both quarterly statements",
            "company FCF is after tax and interest and is not an FCFF/WACC input",
            "quarterly intangibles are omitted, not an official zero",
            "land, office-campus CIP, buyer-protection, and interest bridges are not a statutory capex substitute",
        ],
    }


_PEER_ROLES = frozenset({"design", "foundry", "equipment", "memory", "eda"})
_FORBIDDEN_PEER_FIELDS = frozenset(
    {"market_cap", "pe", "revenue", "ebitda", "price", "net_income", "eps", "shares"}
)


def load_sourced_peers(path: Path) -> dict[str, Any]:
    """Tickers and reasons only. Financials are fetched later, never taken from the file."""
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "sourced_peer_list_v1":
        raise ValueError("sourced peers require schema sourced_peer_list_v1")
    target = str(document["ticker"]).strip().upper()
    role = str(document["role"]).strip().lower()
    rows = document["peers"]
    if role not in _PEER_ROLES:
        raise ValueError(f"peer role {role!r} is not a single business-model role")
    if not isinstance(rows, list) or not 3 <= len(rows) <= 10:
        raise ValueError("a sourced peer list needs 3 to 10 peers")
    tickers: list[str] = []
    reasons: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each peer must be an object")
        forbidden = _FORBIDDEN_PEER_FIELDS.intersection(row)
        if forbidden:
            raise ValueError(f"peer list must not carry financials: {sorted(forbidden)}")
        peer_role = str(row.get("role") or "").strip().lower()
        if peer_role != role:
            raise ValueError("peers must share one business-model role; mixed roles are refused")
        url = str(row.get("source_url") or "")
        if not url.startswith(("https://", "http://")):
            raise ValueError("each peer needs an http(s) source")
        symbol = str(row["ticker"]).strip().upper()
        if symbol == target or symbol in tickers:
            continue
        reason = str(row.get("reason") or "").strip()
        if len(reason) < 8:
            raise ValueError(f"{symbol} needs an explicit reason")
        tickers.append(symbol)
        reasons.append(f"{symbol} ({peer_role}): {reason} [{url}]")
    if len(tickers) < 3:
        raise ValueError("fewer than 3 distinct sourced peers remain")
    return {
        "ticker": target,
        "role": role,
        "tickers": tickers,
        "rationale": "Sourced same-role peers. Financials are not in this list. "
        + " ".join(reasons),
        "financials_included": False,
    }


def _signed_flow(field: dict[str, Any]) -> float:
    magnitude = float(field["value"])
    if field.get("sign") == "cash_outflow" and magnitude > 0:
        return -magnitude
    return magnitude


def _signed_bridge(
    *,
    caption: str,
    role: str,
    fy_end: date,
    fy_value: float,
    current_end: date,
    current_value: float,
    prior_end: date,
    prior_value: float,
    source: str,
    currency: str,
    as_capex_outflow: bool,
) -> dict[str, Any]:
    built = bridge_same_caption(
        caption=caption,
        role=role,
        fy_end=fy_end,
        fy_value=fy_value,
        current_end=current_end,
        current_value=current_value,
        prior_end=prior_end,
        prior_value=prior_value,
        currency=currency,
        source=source,
        as_capex_outflow=as_capex_outflow,
    )
    built["unit"] = "million"
    return built
