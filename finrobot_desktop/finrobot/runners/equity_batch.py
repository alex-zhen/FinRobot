"""Per-ticker equity-research entry. No API key is stored in this module.

On the box, ``DEEPSEEK_API_KEY`` is read from the process environment by
:func:`finrobot.config.deepseek_api_key_from_process` and injected as
``provider_keys={'deepseek': ...}``. A closed input gate does not call the
model and does not emit a target price.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

from finrobot.config import (
    FinRobotSettings,
    ProviderConfig,
    deepseek_api_key_from_process,
    get_settings,
    sec_user_agent_from_process,
)
from finrobot.engine.data.normalize.valuation_input import publication_block_reasons
from finrobot.engine.models.financial import DataProvenance, FinancialData
from finrobot.runners.official_inputs import (
    OfficialTtmCapex,
    period_end_ordinary_share_block,
)
from finrobot.runners.serialize import dump_jsonable

logger = logging.getLogger(__name__)

MODEL_NAME = "deepseek:deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MAX_ATTEMPTS = 2
TIMEOUT_SECONDS = 900

DEEPSEEK_PROVIDER = ProviderConfig(
    id="deepseek",
    label="DeepSeek",
    kind="openai-compatible",
    base_url=DEEPSEEK_BASE_URL,
    models=["deepseek-chat"],
)


def apply_official_ttm_capex(
    financials: FinancialData, built: OfficialTtmCapex
) -> tuple[FinancialData, str | None]:
    """Copy a constructed statutory TTM capex onto the snapshot, or refuse."""
    if not built.ok or built.outflow is None or built.currency is None:
        return financials, built.reason or "official TTM capex was not constructed"
    reporting = financials.reporting_currency.upper()
    if reporting not in {built.currency, "UNKNOWN"}:
        return financials, (
            f"official capex is {built.currency} but the snapshot reports {reporting}"
        )
    updated = financials.model_copy(deep=True)
    updated.income.capital_expenditure = built.outflow
    updated.reporting_currency = built.currency
    updated.ttm_construction = built.construction
    if built.period_ends:
        updated.fiscal_period_end = max(built.period_ends)
        updated.ttm_quarter_ends = list(built.period_ends)
    if updated.provenance is None:
        updated.provenance = DataProvenance(
            provider="official_filing",
            as_of=updated.fiscal_period_end or date.today(),
            period_basis="ttm",
        )
    else:
        updated.provenance.period_basis = "ttm"
        updated.provenance.degraded = [
            item for item in updated.provenance.degraded if item != "period_basis_unknown"
        ]
    return updated, None


def gate_reasons(
    financials: FinancialData,
    *,
    share_as_of: date | None = None,
    price_as_of: date | None = None,
    official_shares: float | None = None,
) -> list[str]:
    """Publication gate plus the period-end share date check. Neither is skipped."""
    reasons = publication_block_reasons(financials)
    if official_shares is not None and share_as_of is not None:
        extra = period_end_ordinary_share_block(
            shares=official_shares,
            share_as_of=share_as_of,
            price=financials.market.current_price,
            price_as_of=price_as_of or date.min,
            market_cap=financials.market.market_cap,
        )
        if extra:
            reasons.append(extra)
    return reasons


def input_record(financials: FinancialData, built: OfficialTtmCapex | None) -> dict[str, Any]:
    provenance = financials.provenance
    return {
        "reporting_currency": financials.reporting_currency,
        "quote_currency": financials.quote_currency,
        "period_basis": None if provenance is None else provenance.period_basis,
        "period_end": None
        if financials.fiscal_period_end is None
        else financials.fiscal_period_end.isoformat(),
        "ttm_construction": financials.ttm_construction,
        "ttm_quarter_ends": [item.isoformat() for item in financials.ttm_quarter_ends],
        "shares_outstanding": financials.market.shares_outstanding,
        "share_basis": financials.share_basis,
        "share_class_scope": financials.share_class_scope,
        "capex_present": financials.income.capital_expenditure is not None,
        "official_capex_sources": [] if built is None else list(built.sources),
        "official_capex_construction": None if built is None else built.construction,
        "fx_conversions": financials.fx_conversions,
    }


def resolved_sec_user_agent() -> str | None:
    """A user-supplied SEC identity, or None. The placeholder is never sent."""
    raw = sec_user_agent_from_process()
    if raw is None:
        return None
    from finrobot.engine.data.providers.edgar_provider import _is_valid_identity

    if not _is_valid_identity(raw):
        logger.warning(
            "FINROBOT_SEC_USER_AGENT is not a usable Name email@domain identity; "
            "EDGAR will not be contacted"
        )
        return None
    return raw


def settings_for_box(api_key: str) -> FinRobotSettings:
    """Inject the key the caller already read. This function does not read the environment."""
    overrides: dict[str, Any] = {}
    identity = resolved_sec_user_agent()
    if identity:
        overrides["sec_user_agent"] = identity
    return get_settings(
        model_name=MODEL_NAME,
        language="zh",
        custom_providers=[DEEPSEEK_PROVIDER],
        provider_keys={"deepseek": api_key},
        **overrides,
    )


def retryable(exc: BaseException) -> bool:
    return isinstance(exc, (TimeoutError, asyncio.TimeoutError))


async def run_research(ticker: str, api_key: str) -> Any:
    from finrobot.sdk import FinRobot

    overrides: dict[str, Any] = {}
    identity = resolved_sec_user_agent()
    if identity:
        overrides["sec_user_agent"] = identity
    else:
        logger.info(
            "SEC EDGAR is not contacted. Set FINROBOT_SEC_USER_AGENT to a real "
            "Name email@domain before filing text is requested. Persisted official "
            "figures stay on the numeric gate."
        )
    agent = FinRobot(
        model=MODEL_NAME,
        language="zh",
        custom_providers=[DEEPSEEK_PROVIDER],
        provider_keys={"deepseek": api_key},
        **overrides,
    )
    try:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return await asyncio.wait_for(agent.aresearch(ticker), timeout=TIMEOUT_SECONDS)
            except (TimeoutError, asyncio.TimeoutError) as exc:
                if attempt == MAX_ATTEMPTS or not retryable(exc):
                    raise
                logger.warning("retrying %s after timeout (%s/%s)", ticker, attempt, MAX_ATTEMPTS)
        raise RuntimeError(f"{ticker} research attempts were exhausted")
    finally:
        await agent.close()


def write_manifest(directory: Path, payload: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_pipeline(directory: Path, result: Any) -> None:
    (directory / "pipeline.json").write_text(
        json.dumps(dump_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def require_key_for_open_gate() -> str:
    key = deepseek_api_key_from_process()
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set; research was not started")
    return key


def manifest_time() -> str:
    return datetime.now().astimezone().isoformat()


async def _amain(tickers: list[str], out: Path, evidence: list[Path]) -> int:
    from finrobot.engine.compute.coordinators.extractor import (
        extract_financial_data,
        normalize_financials_to_usd,
    )
    from finrobot.engine.data.normalize.financials import normalize_financials
    from finrobot.engine.data.normalize.price import normalize_price
    from finrobot.engine.data.providers.yfinance_provider import YFinanceProvider
    from finrobot.engine.data.types import DataType
    from finrobot.runners.evidence_loader import (
        baba_line_records,
        load_finrobot_evidence_capex,
        load_tencent_period_end_shares,
        load_tencent_statutory_capex,
        tencent_same_day_share_record,
    )
    from finrobot.runners.official_inputs import OfficialAmount, construct_official_ttm_capex

    official: list[OfficialAmount] = []
    tencent_shares: tuple[float, date] | None = None
    tencent_price_path: Path | None = None
    supplements: dict[str, Any] = {}
    for path in evidence:
        text = path.read_text(encoding="utf-8")
        if '"capex_cash_ppe_cip_ip"' in text:
            official.extend(load_tencent_statutory_capex(path))
            tencent_shares = load_tencent_period_end_shares(path)
        elif '"baba_statutory_capex_ttm_search_v1"' in text:
            supplements["BABA"] = baba_line_records(path)
        elif '"closing_price_supplement_akshare_sina"' in text:
            tencent_price_path = path
        elif '"issuers"' in text:
            official.extend(load_finrobot_evidence_capex(path))
    provider = YFinanceProvider()
    blocked = 0
    for ticker in tickers:
        directory = out / ticker.replace("/", "_")
        try:
            financials_raw = await provider.fetch(ticker, DataType.FINANCIALS)
            price_raw = await provider.fetch(ticker, DataType.PRICE)
            extracted = extract_financial_data(
                normalize_financials(financials_raw), normalize_price(price_raw)
            )
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, LookupError) as exc:
            write_manifest(
                directory,
                {
                    "ticker": ticker,
                    "research_ran": False,
                    "error": type(exc).__name__,
                    "reasons": [f"input_error: {type(exc).__name__}"],
                },
            )
            blocked += 1
            continue
        built = construct_official_ttm_capex(official, extracted.ticker) if official else None
        official_note = None if built is None or built.ok else built.reason
        if built is not None and built.ok:
            extracted, apply_reason = apply_official_ttm_capex(extracted, built)
            if apply_reason:
                official_note = apply_reason
                built = None
        fx_error: str | None = None
        try:
            extracted = await normalize_financials_to_usd(extracted, fmp_api_key=None)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, LookupError) as exc:
            fx_error = f"fx_missing: {type(exc).__name__}"
        price_as_of = price_raw.timestamp.date() if hasattr(price_raw, "timestamp") else None
        official_shares = None
        share_as_of = None
        if ticker.upper() in {"0700.HK", "TCEHY"} and tencent_shares is not None:
            official_shares, share_as_of = tencent_shares
        reasons = gate_reasons(
            extracted,
            share_as_of=share_as_of,
            price_as_of=price_as_of,
            official_shares=official_shares,
        )
        if fx_error:
            reasons.append(fx_error)
        deduped: list[str] = []
        for reason in reasons:
            if reason not in deduped:
                deduped.append(reason)
        supplement = supplements.get(ticker.upper())
        historical_bridge = None
        if ticker.upper() in {"0700.HK", "TCEHY"} and tencent_price_path is not None:
            historical_bridge = tencent_same_day_share_record(
                tencent_price_path,
                valuation_as_of=price_as_of or date.max,
            )
        payload = {
            "ticker": ticker,
            "recorded_at": manifest_time(),
            "research_ran": False,
            "sec_edgar_contacted": resolved_sec_user_agent() is not None,
            "inputs": input_record(extracted, built if built is not None and built.ok else None),
            "official_note": official_note,
            "line_records": supplement,
            "historical_share_bridge": historical_bridge,
            "reasons": deduped,
        }
        if deduped:
            write_manifest(directory, payload)
            logger.info("%s refused: %s", ticker, "; ".join(deduped))
            blocked += 1
            continue
        try:
            key = require_key_for_open_gate()
        except RuntimeError as exc:
            payload["reasons"] = [str(exc)]
            write_manifest(directory, payload)
            blocked += 1
            continue
        result = await run_research(ticker, key)
        payload["research_ran"] = True
        write_manifest(directory, payload)
        write_pipeline(directory, result)
    return 1 if blocked else 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Per-ticker equity-research gate and run.")
    parser.add_argument("tickers", nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return asyncio.run(_amain(args.tickers, args.out, args.evidence))


if __name__ == "__main__":
    raise SystemExit(main())
