"""Valuation-input caliber: TTM window, capex sign, FX field policy, share basis.

Pure functions. No network, no hardcoded spot rate, and no use of a prior
audit's printed figures as if they were filings. A publishable DCF is a
current fair value built only from a window and a currency basis this module
can name. Anything else is a refusal with a specific reason.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from finrobot.engine.data.ticker import validate_ticker
from finrobot.engine.primitives.book_value import SHARES_PRICE_CONSISTENCY_TOL

# A fiscal quarter is about 13 weeks (91 days). 80–120 admits a 12-week or
# 16-week (53-week-year) quarter and rejects a monthly stub or a half-year.
_QUARTER_GAP_DAYS = (80, 120)
# A fiscal half is about 182 days. The band is disjoint from the quarter band.
_HALF_GAP_DAYS = (150, 210)

# Spot translation policy. Flow and stock items use the caller-supplied spot
# for their own currency. Per-share amounts use that same currency rate and are
# never multiplied by an ADS or split ratio (that ratio is a share-count
# adjustment, applied separately). Quote items (price, market cap) use the
# quote currency's rate only. Margins, share counts and beta are not amounts.
# This is today's spot, not an IAS 21 average and not a stored official fixing.
FX_POLICY = (
    "flow and stock amounts scale by the reporting-currency spot; "
    "per-share amounts scale by that same spot and are not adjusted by an "
    "ADS or split ratio here; quote amounts (price, market cap) scale only by "
    "the quote-currency spot; margins, share counts and beta are unchanged. "
    "The rate is caller-supplied spot, not a hardcoded fixing and not a "
    "period-average rate. A snapshot already expressed in the target currency "
    "is not converted again."
)

# Fields the USD normalizer must scale together. Leaving one behind is the
# mixed-currency bug (revenue in USD, capex or operating EBITDA still local).
FX_REPORTING_FIELDS = (
    "income.revenue",
    "income.ebitda",
    "income.net_income",
    "income.operating_income",
    "income.depreciation_amortization",
    "income.rd_expense",
    "income.sga_expense",
    "income.interest_expense",
    "income.income_tax_expense",
    "income.capital_expenditure",
    "valuation.ebitda_operating",
    "valuation.ebitda_reported",
    "valuation.book_value_per_share",
    "balance.total_debt",
    "balance.total_cash",
    "balance.preferred_stock",
    "balance.noncontrolling_interest",
)
FX_QUOTE_FIELDS = (
    "market.market_cap",
    "market.current_price",
    "market.price_52w_high",
    "market.price_52w_low",
)

_NAME_SUFFIXES = frozenset(
    {
        "corp",
        "corporation",
        "inc",
        "incorporated",
        "co",
        "company",
        "companies",
        "ltd",
        "limited",
        "plc",
        "llc",
        "lp",
        "group",
        "holdings",
        "holding",
        "the",
        "and",
    }
)
_NAME_SPLIT = re.compile(r"[^a-z0-9]+")

Construction = Literal["four_quarters", "two_halves", "fy_ytd_bridge"]


@dataclass(frozen=True)
class StatementWindow:
    """Rows that may be summed into one trailing twelve months, or a refusal."""

    construction: Construction | None
    indexes: tuple[int, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class FlowColumn:
    """One reported flow column. ``kind`` is explicit; it is never guessed
    from the size of the number."""

    period_end: date
    kind: Literal["quarter", "half", "fy", "ytd"]
    currency: str
    source: str
    values: Mapping[str, float | None]


@dataclass(frozen=True)
class TtmFlows:
    status: Literal["complete", "incomplete"]
    construction: Construction | None
    period_ends: tuple[date, ...]
    currency: str | None
    values: dict[str, float | None]
    capital_expenditure_outflow: float | None
    reasons: tuple[str, ...]
    sources: tuple[str, ...]


@dataclass(frozen=True)
class EntityResolution:
    status: Literal["ticker", "unique_name", "refused"]
    ticker: str | None
    reasons: tuple[str, ...]


def capex_outflow(signed: float | None) -> tuple[float | None, str | None]:
    """Map a cash-flow capital-expenditure line to a positive outflow.

    A negative amount is a cash outflow (the usual statement convention) and
    becomes its magnitude. Zero stays zero. A positive amount is not turned
    into an outflow: it may be a net inflow or a different field, and taking
    the absolute value would publish that as spending.
    """
    if signed is None:
        return None, None
    try:
        value = float(signed)
    except (TypeError, ValueError):
        return None, "capex_sign_unverified: capital expenditure is not a number"
    if not math.isfinite(value):
        return None, "capex_sign_unverified: capital expenditure is not finite"
    if value < 0:
        return -value, None
    if value == 0:
        return 0.0, None
    return (
        None,
        (
            "capex_sign_unverified: capital expenditure is non-negative, so it was "
            "not treated as an outflow"
        ),
    )


def _gap_days(newer: date, older: date) -> int:
    return (newer - older).days


def select_statement_window(period_ends: Sequence[date | None]) -> StatementWindow:
    """Choose a summable trailing-twelve-month window, or refuse.

    Accepts four period-ends whose consecutive gaps are fiscal quarters, or
    two period-ends whose gap is a fiscal half when every observed gap is a
    half (a semi-annual filer). Duplicate, undated, monthly, or gapped rows
    are not summed.
    """
    if not period_ends:
        return StatementWindow(None, (), ("ttm_window_unverified: no period ends",))
    if any(d is None for d in period_ends):
        return StatementWindow(None, (), ("ttm_window_unverified: a row has no period end",))
    dates = [d for d in period_ends if d is not None]
    if len(set(dates)) != len(dates):
        return StatementWindow(
            None,
            (),
            ("ttm_overlap: duplicate period end; refusing to sum restated rows",),
        )
    order = sorted(range(len(dates)), key=lambda i: dates[i], reverse=True)
    ordered = [dates[i] for i in order]
    gaps = [_gap_days(ordered[i], ordered[i + 1]) for i in range(len(ordered) - 1)]

    if len(order) >= 4 and all(_QUARTER_GAP_DAYS[0] <= g <= _QUARTER_GAP_DAYS[1] for g in gaps[:3]):
        return StatementWindow("four_quarters", tuple(order[:4]), ())

    if len(order) >= 2 and gaps and all(_HALF_GAP_DAYS[0] <= g <= _HALF_GAP_DAYS[1] for g in gaps):
        return StatementWindow("two_halves", tuple(order[:2]), ())

    return StatementWindow(
        None,
        (),
        (
            "ttm_window_unverified: period ends "
            + ", ".join(d.isoformat() for d in ordered)
            + f" gaps {gaps} are not four non-overlapping quarters or two "
            "non-overlapping halves",
        ),
    )


def _same_currency(columns: Sequence[FlowColumn]) -> str | None:
    currencies = {c.currency.strip().upper() for c in columns}
    if len(currencies) != 1:
        return None
    ccy = next(iter(currencies))
    return ccy if len(ccy) == 3 and ccy.isalpha() else None


def _subtract_flows(
    columns: Sequence[FlowColumn], signs: Sequence[float]
) -> dict[str, float | None]:
    keys: set[str] = set()
    for column in columns:
        keys.update(column.values)
    out: dict[str, float | None] = {}
    for key in keys:
        total = 0.0
        for column, sign in zip(columns, signs):
            raw = column.values.get(key)
            if raw is None:
                total_missing = True
                break
            try:
                number = float(raw)
            except (TypeError, ValueError):
                total_missing = True
                break
            if not math.isfinite(number):
                total_missing = True
                break
            total += sign * number
        else:
            total_missing = False
        out[key] = None if total_missing else total
    return out


def bridge_fy_ytd(fy: FlowColumn, current_ytd: FlowColumn, prior_ytd: FlowColumn) -> TtmFlows:
    """TTM = completed fiscal year + current YTD − prior-year YTD.

    All three columns must be explicitly tagged, share one currency, and the
    two YTD columns must end on the same fiscal stub a year apart. The prior
    YTD must fall inside the completed year. Missing fields stay missing.
    """
    reasons: list[str] = []
    if (fy.kind, current_ytd.kind, prior_ytd.kind) != ("fy", "ytd", "ytd"):
        reasons.append("ttm_ytd_unverified: columns are not one FY and two YTD statements")
    currency = _same_currency((fy, current_ytd, prior_ytd))
    if currency is None:
        reasons.append("ttm_ytd_unverified: FY and YTD columns are not in one currency")
    stub_gap = (current_ytd.period_end - prior_ytd.period_end).days
    if not 360 <= stub_gap <= 370:
        reasons.append(
            "ttm_ytd_unverified: current and prior YTD do not end one year apart "
            f"({prior_ytd.period_end.isoformat()} vs {current_ytd.period_end.isoformat()})"
        )
    if not (prior_ytd.period_end <= fy.period_end <= current_ytd.period_end):
        reasons.append(
            "ttm_ytd_unverified: the completed year does not sit between the prior "
            "YTD and the current YTD"
        )
    if reasons:
        return TtmFlows("incomplete", None, (), currency, {}, None, tuple(reasons), ())
    values = _subtract_flows((fy, current_ytd, prior_ytd), (1.0, 1.0, -1.0))
    signed_capex = values.get("capital_expenditure")
    outflow, note = capex_outflow(signed_capex)
    extra = (note,) if note else ()
    return TtmFlows(
        "complete",
        "fy_ytd_bridge",
        (fy.period_end, prior_ytd.period_end, current_ytd.period_end),
        currency,
        values,
        outflow,
        extra,
        (fy.source, prior_ytd.source, current_ytd.source),
    )


def split_conflict(
    old_shares: float,
    new_shares: float,
    old_price: float,
    new_price: float,
    *,
    tolerance: float = 0.15,
) -> str | None:
    """Return a reason when a share-count jump is not matched by the inverse price move.

    A split or ADS ratio changes the share count and the per-share price together.
    The market cap (price × shares) should stay in the same neighborhood. A jump
    in only one of them mixes a pre-split count with a post-split price.
    """
    if min(old_shares, new_shares, old_price, new_price) <= 0:
        return "split_unreconciled: share count or price is not positive"
    if not all(math.isfinite(v) for v in (old_shares, new_shares, old_price, new_price)):
        return "split_unreconciled: share count or price is not finite"
    share_factor = new_shares / old_shares
    if 1 / 1.5 < share_factor < 1.5:
        return None
    price_factor = new_price / old_price
    if abs(price_factor * share_factor - 1.0) > tolerance:
        return (
            "split_unreconciled: share count moved by "
            f"{share_factor:.2f}x while the price moved by {price_factor:.2f}x"
        )
    return None


def ads_quote_is_consistent(
    ordinary_shares: float,
    ordinary_per_ads: float,
    price_per_ads: float,
    market_cap: float,
    *,
    tolerance: float = SHARES_PRICE_CONSISTENCY_TOL,
) -> bool:
    """True when ordinary shares ÷ ADS ratio reproduces price × shares ≈ market cap."""
    if ordinary_per_ads <= 0 or ordinary_shares <= 0 or price_per_ads <= 0 or market_cap <= 0:
        return False
    ads = ordinary_shares / ordinary_per_ads
    implied_cap = price_per_ads * ads
    return abs(implied_cap - market_cap) / market_cap <= tolerance


def _normalize_legal_name(name: str) -> str:
    tokens = [t for t in _NAME_SPLIT.split(name.lower()) if t and t not in _NAME_SUFFIXES]
    return " ".join(tokens)


def resolve_listed_entity(
    query: str, catalog: Sequence[tuple[str, str]] | None = None
) -> EntityResolution:
    """Resolve a research subject without fuzzy ticker substitution.

    A catalog entry is ``(ticker, legal_name)`` from a source the caller loaded
    (for example the SEC company list). A name matches only that full normalized
    legal name. A ticker-shaped query is that ticker, never a different symbol
    whose letters are close. Zero matches and more than one match are refusals.
    """
    text = (query or "").strip()
    if not text:
        return EntityResolution("refused", None, ("entity_unspecified: empty query",))
    entries = list(catalog or [])
    wanted = _normalize_legal_name(text)
    by_name = [
        ticker for ticker, legal in entries if wanted and _normalize_legal_name(legal) == wanted
    ]
    try:
        syntax = validate_ticker(text)
    except ValueError:
        syntax = None
    if len(by_name) > 1:
        return EntityResolution(
            "refused",
            None,
            ("entity_ambiguous: " + text + " matches " + ", ".join(sorted(set(by_name))),),
        )
    if len(by_name) == 1:
        ticker = by_name[0]
        if syntax is not None and syntax != ticker.upper():
            return EntityResolution(
                "refused",
                None,
                (
                    (
                        f"entity_mismatch: query {text} looks like ticker {syntax} "
                        f"but the legal name belongs to {ticker}"
                    ),
                ),
            )
        return EntityResolution("unique_name", ticker, ())
    if syntax is not None:
        known = {ticker.upper() for ticker, _ in entries}
        if entries and syntax not in known:
            return EntityResolution(
                "refused",
                None,
                (
                    (
                        f"entity_not_listed: {syntax} is not in the supplied catalog "
                        "and was not substituted from a similar name"
                    ),
                ),
            )
        note = () if entries else ("entity_ticker_only: issuer was not checked against a catalog",)
        return EntityResolution("ticker", syntax, note)
    return EntityResolution(
        "refused",
        None,
        (f"entity_not_listed: no catalog legal name equals {text!r}",),
    )


def _iso(code: str | None) -> str:
    text = (code or "").strip().upper()
    if len(text) == 3 and text.isalpha():
        return text
    return ""


def _has_fx_rate(record: Mapping[str, Any]) -> bool:
    for key in ("reporting_rate", "quote_rate"):
        raw = record.get(key)
        if isinstance(raw, (int, float)) and math.isfinite(float(raw)) and float(raw) > 0:
            return True
    return False


def publication_block_reasons(financials: Any) -> list[str]:
    """Reasons a DCF and a buy/sell recommendation must not be published.

    ``financials`` is a ``FinancialData`` after the USD normalization step.
    The checks are the currency tags, the FX record, the TTM window, and the
    share basis. They do not invent a rate or a quarter.
    """
    reasons: list[str] = []
    ticker = str(getattr(financials, "ticker", "") or "")
    reporting = _iso(getattr(financials, "reporting_currency", None))
    quote = _iso(getattr(financials, "quote_currency", None))
    if not reporting or not quote:
        reasons.append(
            "currency_unspecified: reporting or quote currency is missing, so no "
            "target price is computed"
        )
    elif reporting != quote:
        reasons.append(
            f"currency_not_unified: reporting {reporting} and quote {quote} are still "
            "different, so statement lines and the market price are not one currency"
        )

    provenance = getattr(financials, "provenance", None)
    degraded = list(getattr(provenance, "degraded", None) or [])
    records = list(getattr(financials, "fx_conversions", None) or [])
    converted = any(_has_fx_rate(r) for r in records if isinstance(r, Mapping))
    if "fx_unavailable" in degraded and not converted:
        reasons.append("fx_missing: a required exchange rate was not available")
    if reporting and quote and reporting == quote == "USD" and records:
        for record in records:
            if not isinstance(record, Mapping) or record.get("scaler") != "financialdata":
                continue
            scaled = set(record.get("scaled_fields") or [])
            present = {
                "income.capital_expenditure": getattr(
                    getattr(financials, "income", None), "capital_expenditure", None
                ),
                "valuation.ebitda_operating": getattr(
                    getattr(financials, "valuation", None), "ebitda_operating", None
                ),
                "valuation.book_value_per_share": getattr(
                    getattr(financials, "valuation", None), "book_value_per_share", None
                ),
                "income.ebitda": getattr(getattr(financials, "income", None), "ebitda", None),
                "income.revenue": getattr(getattr(financials, "income", None), "revenue", None),
            }
            for field, value in present.items():
                if value is not None and field not in scaled:
                    reasons.append(
                        f"fx_field_not_converted: {field} was left in the source currency "
                        "while other amounts were converted"
                    )

    basis = getattr(provenance, "period_basis", None) if provenance is not None else None
    if (
        provenance is None
        or basis in (None, "", "unverified")
        or "period_basis_unknown" in degraded
    ):
        reasons.append(
            "period_unspecified: the statement period is not a verified TTM, annual, "
            "or quarterly figure"
        )
    elif basis == "ttm":
        ends = list(getattr(financials, "ttm_quarter_ends", None) or [])
        construction = getattr(financials, "ttm_construction", None)
        if construction == "fy_ytd_bridge":
            if getattr(financials, "fiscal_period_end", None) is None or len(ends) < 3:
                reasons.append(
                    "period_end_missing: a FY+YTD bridge needs the completed year and "
                    "both YTD period ends"
                )
        else:
            window = select_statement_window(ends)
            if window.construction is None:
                reasons.extend(window.reasons or ("ttm_window_unverified: TTM dates are missing",))
            elif construction not in (None, window.construction):
                reasons.append(
                    f"ttm_window_unverified: labeled {construction} but the dates are "
                    f"{window.construction}"
                )
            if getattr(financials, "fiscal_period_end", None) is None:
                reasons.append("period_end_missing: TTM has no fiscal period end")
    elif basis == "annual":
        if getattr(financials, "fiscal_period_end", None) is None:
            reasons.append("period_end_missing: the annual statement has no period end")
    else:
        reasons.append(
            "period_not_twelve_months: a quarterly or unrecognized statement is not a "
            "trailing twelve months and is not used as a formal valuation base"
        )

    market = getattr(financials, "market", None)
    price = getattr(market, "current_price", None)
    shares = getattr(market, "shares_outstanding", None)
    cap = getattr(market, "market_cap", None)
    share_basis = getattr(financials, "share_basis", None)
    if share_basis == "derived_missing":
        reasons.append(
            "shares_unverified: the share count was implied by price and market cap "
            "and was not independently reported"
        )
    elif share_basis == "unresolved":
        reasons.append(
            "shares_price_market_cap_inconsistent: the reported share count does not "
            "match price times shares on the quote, and no ADS ratio reconciles them"
        )
    elif (
        isinstance(price, (int, float))
        and isinstance(shares, (int, float))
        and isinstance(cap, (int, float))
        and price > 0
        and shares > 0
        and cap > 0
    ):
        gap = abs(price * shares - cap) / cap
        if gap > SHARES_PRICE_CONSISTENCY_TOL:
            reasons.append(
                "shares_price_market_cap_inconsistent: price × shares differs from "
                f"market cap by {gap:.1%}"
            )
    else:
        reasons.append("shares_unverified: price, shares, or market cap is missing")

    capex = getattr(getattr(financials, "income", None), "capital_expenditure", None)
    if capex is None:
        reasons.append(
            "capex_missing: trailing capital expenditure is absent, so a publishable DCF is refused"
        )
    elif not isinstance(capex, (int, float)) or not math.isfinite(float(capex)) or float(capex) < 0:
        reasons.append("capex_sign_unverified: capital expenditure is not a non-negative outflow")

    scope = getattr(financials, "share_class_scope", None)
    if "." in ticker and scope not in {"single_listing", "ads_reconciled"}:
        reasons.append(
            "share_class_scope_unverified: "
            f"{ticker} uses an exchange suffix and the snapshot does not say whether "
            "the share count is this listing only or the whole company (A/H or "
            "another class). Per-share DCF is refused"
        )

    observation = getattr(financials, "split_observation", None)
    if isinstance(observation, Mapping):
        try:
            conflict = split_conflict(
                float(observation["old_shares"]),
                float(observation["new_shares"]),
                float(observation["old_price"]),
                float(observation["new_price"]),
            )
        except (KeyError, TypeError, ValueError):
            conflict = "split_unreconciled: the split observation is incomplete"
        if conflict:
            reasons.append(conflict)
    return reasons
