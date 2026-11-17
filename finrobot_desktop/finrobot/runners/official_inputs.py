"""Traceable official-statement amounts. They never bypass the publication gate.

A record is one line from one filing: source URL, statement label, report
period, currency, and unit. A full-year or half-year capital-expenditure
amount is not a trailing twelve months. TTM capex is built only from four
non-overlapping quarters or from an explicit full year plus current YTD minus
the prior YTD. The publication gate still runs on the resulting snapshot.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from finrobot.engine.data.normalize.valuation_input import (
    FlowColumn,
    bridge_fy_ytd,
    capex_outflow,
    select_statement_window,
)

LineRole = Literal[
    "statutory_capex",
    "management_capex",
    "company_fcf",
    "fcf_recon_ppe",
    "self_calculated_fcf",
]
LineItem = Literal["capital_expenditure"]
PeriodType = Literal["quarter", "half", "fy", "ytd"]
Unit = Literal["ones", "thousand", "million", "billion"]
_UNIT_SCALE = {"ones": 1.0, "thousand": 1e3, "million": 1e6, "billion": 1e9}


class OfficialAmount(BaseModel):
    """One sourced statement line. ``value`` keeps the statement sign."""

    ticker: str
    source_url: str = Field(min_length=8)
    source_kind: str = Field(min_length=2)
    statement_line: str = Field(min_length=2)
    line_item: LineItem
    # Only statutory_capex can become TTM capex. A collection flag such as
    # usable_as_finrobot_input is not this role and is not read here.
    line_role: LineRole = "statutory_capex"
    period_start: date
    period_end: date
    period_type: PeriodType
    currency: str
    unit: Unit
    value: float

    @field_validator("ticker", "currency")
    @classmethod
    def _upper(cls, raw: str) -> str:
        return raw.strip().upper()

    @field_validator("currency")
    @classmethod
    def _iso(cls, raw: str) -> str:
        if len(raw) != 3 or not raw.isalpha():
            raise ValueError("currency must be a 3-letter ISO code")
        return raw

    @field_validator("source_url")
    @classmethod
    def _url(cls, raw: str) -> str:
        text = raw.strip()
        if not text.startswith(("https://", "http://")):
            raise ValueError("source_url must be an http(s) URL")
        return text


class OfficialShareScope(BaseModel):
    """A sourced statement of which share line a listing uses.

    Setting the scope does not change the share count and does not skip the
    price × shares check.
    """

    ticker: str
    source_url: str = Field(min_length=8)
    source_kind: str = Field(min_length=2)
    period_end: date
    scope: Literal["single_listing", "ads_reconciled"]
    note: str = Field(min_length=8)

    @field_validator("ticker")
    @classmethod
    def _upper(cls, raw: str) -> str:
        return raw.strip().upper()

    @field_validator("source_url")
    @classmethod
    def _url(cls, raw: str) -> str:
        text = raw.strip()
        if not text.startswith(("https://", "http://")):
            raise ValueError("source_url must be an http(s) URL")
        return text


class OfficialSupplement(BaseModel):
    amounts: list[OfficialAmount] = Field(default_factory=list)
    share_scopes: list[OfficialShareScope] = Field(default_factory=list)


class OfficialTtmCapex:
    """A constructed trailing capex outflow, or a refusal."""

    def __init__(
        self,
        *,
        outflow: float | None,
        construction: str | None,
        currency: str | None,
        sources: tuple[str, ...],
        reason: str | None,
    ) -> None:
        self.outflow = outflow
        self.construction = construction
        self.currency = currency
        self.sources = sources
        self.reason = reason
        self.period_ends: tuple[date, ...] = ()

    @property
    def ok(self) -> bool:
        return self.outflow is not None and self.reason is None


def _scaled(amount: OfficialAmount) -> float:
    return float(amount.value) * _UNIT_SCALE[amount.unit]


def _same_currency(rows: list[OfficialAmount]) -> str | None:
    currencies = {row.currency for row in rows}
    if len(currencies) != 1:
        return None
    return next(iter(currencies))


def construct_official_ttm_capex(records: list[OfficialAmount], ticker: str) -> OfficialTtmCapex:
    """Build TTM capex for ``ticker`` or explain why the records are not enough."""
    rows = [
        row
        for row in records
        if row.ticker == ticker.strip().upper() and row.line_role == "statutory_capex"
    ]
    if not rows:
        return OfficialTtmCapex(
            outflow=None,
            construction=None,
            currency=None,
            sources=(),
            reason="no official capital-expenditure rows for this ticker",
        )
    quarters = construct_from_quarters(rows)
    bridge = construct_from_ytd_bridge(rows)
    if quarters.ok and bridge.ok:
        return OfficialTtmCapex(
            outflow=None,
            construction=None,
            currency=None,
            sources=(),
            reason=(
                "both a four-quarter window and a FY+YTD bridge are present; "
                "supply only one construction"
            ),
        )
    if quarters.ok:
        return quarters
    if bridge.ok:
        return bridge
    if any(row.period_type in {"fy", "half"} for row in rows) and not quarters.ok:
        return OfficialTtmCapex(
            outflow=None,
            construction=None,
            currency=_same_currency(rows),
            sources=tuple(row.source_url for row in rows),
            reason=(
                "a full-year or half-year capital-expenditure amount is not a "
                "trailing twelve months and was not substituted for TTM capex. "
                f"quarter window: {quarters.reason}. bridge: {bridge.reason}"
            ),
        )
    return quarters if quarters.reason else bridge


def construct_from_quarters(rows: list[OfficialAmount]) -> OfficialTtmCapex:
    quarters = [row for row in rows if row.period_type == "quarter"]
    if len(quarters) < 4:
        return OfficialTtmCapex(
            outflow=None,
            construction=None,
            currency=_same_currency(quarters) if quarters else None,
            sources=tuple(row.source_url for row in quarters),
            reason="fewer than four quarterly capital-expenditure rows",
        )
    currency = _same_currency(quarters)
    if currency is None:
        return OfficialTtmCapex(
            outflow=None,
            construction=None,
            currency=None,
            sources=tuple(row.source_url for row in quarters),
            reason="quarterly capital-expenditure rows are not in one currency",
        )
    ends = [row.period_end for row in quarters]
    window = select_statement_window(ends)
    if window.construction != "four_quarters":
        return OfficialTtmCapex(
            outflow=None,
            construction=None,
            currency=currency,
            sources=tuple(row.source_url for row in quarters),
            reason=window.reasons[0] if window.reasons else "quarter dates are not a TTM window",
        )
    chosen = [quarters[index] for index in window.indexes]
    signed = sum(_scaled(row) for row in chosen)
    outflow, note = capex_outflow(signed)
    if note or outflow is None:
        return OfficialTtmCapex(
            outflow=None,
            construction=None,
            currency=currency,
            sources=tuple(row.source_url for row in chosen),
            reason=note or "capex sign was not an outflow",
        )
    built = OfficialTtmCapex(
        outflow=outflow,
        construction="four_quarters",
        currency=currency,
        sources=tuple(row.source_url for row in chosen),
        reason=None,
    )
    built.period_ends = tuple(row.period_end for row in chosen)
    return built


def construct_from_ytd_bridge(rows: list[OfficialAmount]) -> OfficialTtmCapex:
    fys = [row for row in rows if row.period_type == "fy"]
    ytds = [row for row in rows if row.period_type == "ytd"]
    matches: list[tuple[OfficialAmount, OfficialAmount, OfficialAmount, float]] = []
    for fy in fys:
        for current in ytds:
            for prior in ytds:
                if prior.period_end >= current.period_end:
                    continue
                currency = _same_currency([fy, current, prior])
                if currency is None:
                    continue
                built = bridge_fy_ytd(
                    FlowColumn(
                        fy.period_end,
                        "fy",
                        currency,
                        fy.source_url,
                        {"capital_expenditure": _scaled(fy)},
                    ),
                    FlowColumn(
                        current.period_end,
                        "ytd",
                        currency,
                        current.source_url,
                        {"capital_expenditure": _scaled(current)},
                    ),
                    FlowColumn(
                        prior.period_end,
                        "ytd",
                        currency,
                        prior.source_url,
                        {"capital_expenditure": _scaled(prior)},
                    ),
                )
                if built.status != "complete" or built.capital_expenditure_outflow is None:
                    continue
                matches.append((fy, current, prior, built.capital_expenditure_outflow))
    if len(matches) != 1:
        return OfficialTtmCapex(
            outflow=None,
            construction=None,
            currency=None,
            sources=tuple(row.source_url for row in (*fys, *ytds)),
            reason=(
                "FY+YTD bridge needs exactly one completed year and the current "
                f"and prior YTD in one currency; found {len(matches)} matches"
            ),
        )
    fy, current, prior, outflow = matches[0]
    built_out = OfficialTtmCapex(
        outflow=outflow,
        construction="fy_ytd_bridge",
        currency=fy.currency,
        sources=(fy.source_url, prior.source_url, current.source_url),
        reason=None,
    )
    built_out.period_ends = (fy.period_end, prior.period_end, current.period_end)
    return built_out


def label_self_calculated_ocf_minus_capex(
    operating_cash_flow: float, capex_outflow: float
) -> dict[str, object]:
    """OCF minus statutory capex, explicitly not a company-defined free-cash-flow line."""
    return {
        "label": "self_calculated_ocf_minus_statutory_capex",
        "company_defined": False,
        "value": operating_cash_flow - capex_outflow,
    }


def period_end_ordinary_share_block(
    *,
    shares: float,
    share_as_of: date,
    price: float,
    price_as_of: date,
    market_cap: float | None,
) -> str | None:
    """Period-end ordinary shares are usable only with that same day's price.

    Weighted-average shares are not accepted here, and a later quote is not a
    reconciliation of the period-end count. This does not skip the publication
    gate's price × shares check.
    """
    if shares <= 0 or price <= 0:
        return "share_price_date_unmatched: share count or price is not positive"
    if price_as_of != share_as_of:
        return (
            "share_price_date_unmatched: period-end ordinary shares are dated "
            f"{share_as_of.isoformat()} but the price is dated {price_as_of.isoformat()}"
        )
    if market_cap is None or market_cap <= 0:
        return "share_price_date_unmatched: no market cap on the share-count date"
    gap = abs(price * shares - market_cap) / market_cap
    if gap > 0.10:
        return (
            "shares_price_market_cap_inconsistent: same-day price × period-end "
            f"ordinary shares differs from market cap by {gap:.1%}"
        )
    return None
