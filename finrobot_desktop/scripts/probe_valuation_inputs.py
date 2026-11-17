"""Keyless valuation-input probe. It does not run a research report.

Official inputs required before a publishable DCF or buy/sell label:

- Reporting currency and quote currency, both ISO 4217. If they differ, a
  caller-supplied spot for each hop. Do not reuse the quote currency's rate
  on statement lines, and do not convert a snapshot that is already in the
  target currency.
- A trailing window that is one of: four non-overlapping fiscal quarters,
  two non-overlapping fiscal halves, or an explicitly tagged full year plus
  current year-to-date minus the same period's prior year-to-date. Each
  selected column needs a period end. A column with no revenue is not summed.
- Capital expenditure as a verified outflow. A negative cash-flow amount is
  the outflow. A missing or non-negative amount is not turned into spending.
- Share count, price, and market cap in one currency and within 10 percent.
  An exchange-suffixed ticker also needs ``share_class_scope`` of
  ``single_listing`` or ``ads_reconciled``. A split must move the price by
  about the inverse of the share-count change.
- A research subject is an exact legal name or that ticker. A nearby symbol
  is not substituted.

This entry fetches yfinance when tickers are passed. It does not read API
keys. A full equity-research report still needs the application's existing
FMP key and model provider key; this script never prints a target price.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime

from finrobot.engine.data.normalize.valuation_input import publication_block_reasons
from finrobot.engine.models.financial import (
    BalanceSheet,
    FinancialData,
    IncomeStatement,
    MarketData,
)


def _refused_sample() -> FinancialData:
    """Synthetic snapshot with an unknown period and no capex. Not a filing."""
    return FinancialData(
        ticker="SYN",
        timestamp=datetime(2026, 4, 1, tzinfo=UTC),
        income=IncomeStatement(revenue=100.0),
        balance=BalanceSheet(),
        market=MarketData(market_cap=1_000.0, shares_outstanding=10.0, current_price=100.0),
        reporting_currency="UNKNOWN",
        quote_currency="USD",
    )


def self_check() -> int:
    reasons = publication_block_reasons(_refused_sample())
    codes = {item.split(":", 1)[0] for item in reasons}
    required = {"currency_unspecified", "period_unspecified", "capex_missing"}
    missing = required - codes
    if missing:
        print("self-check failed, missing blocks:", ", ".join(sorted(missing)))
        return 1
    print("self-check refused a synthetic snapshot with unknown currency, period, and capex")
    for reason in reasons:
        print(" -", reason)
    print("no target price emitted")
    return 0


async def _probe(tickers: list[str]) -> int:
    from finrobot.engine.compute.coordinators.extractor import (
        extract_financial_data,
        normalize_financials_to_usd,
    )
    from finrobot.engine.data.normalize.financials import normalize_financials
    from finrobot.engine.data.normalize.price import normalize_price
    from finrobot.engine.data.providers.yfinance_provider import YFinanceProvider
    from finrobot.engine.data.types import DataType

    provider = YFinanceProvider()
    blocked = 0
    for ticker in tickers:
        print(f"=== {ticker} ===")
        try:
            financials = await provider.fetch(ticker, DataType.FINANCIALS)
            price = await provider.fetch(ticker, DataType.PRICE)
            extracted = extract_financial_data(
                normalize_financials(financials), normalize_price(price)
            )
            extracted = await normalize_financials_to_usd(extracted, fmp_api_key=None)
        except Exception as exc:
            print("input_error", type(exc).__name__)
            print("no target price emitted")
            blocked += 1
            continue
        reasons = publication_block_reasons(extracted)
        print(
            "currencies",
            extracted.reporting_currency,
            extracted.quote_currency,
            "period",
            None if extracted.provenance is None else extracted.provenance.period_basis,
            "capex_present",
            extracted.income.capital_expenditure is not None,
        )
        if reasons:
            blocked += 1
            print("refused")
            for reason in reasons:
                print(" -", reason)
            print("no target price emitted")
        else:
            print("input gate open")
            print("full equity-research report not run; no model key is used here")
            print("no target price emitted")
    return 1 if blocked else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="*", help="Tickers to probe with yfinance")
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Refuse a synthetic snapshot offline and emit no price",
    )
    args = parser.parse_args(argv)
    if args.self_check:
        return self_check()
    if not args.tickers:
        parser.print_help()
        return 2
    return asyncio.run(_probe(args.tickers))


if __name__ == "__main__":
    sys.exit(main())
