"""Offline regression for valuation-input currency, period, capex, and share basis.

Every figure in this file is synthetic. Nothing here is a filing, a market
price, or a number copied from an earlier audit.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from unittest.mock import MagicMock

import pytest

from finrobot.engine.compute.operators.fx_normalize import normalize_financialdata_to_usd
from finrobot.engine.data.normalize.contracts import DEGRADED_FX_UNAVAILABLE
from finrobot.engine.data.normalize.valuation_input import (
    FlowColumn,
    ads_quote_is_consistent,
    bridge_fy_ytd,
    capex_outflow,
    publication_block_reasons,
    resolve_listed_entity,
    select_statement_window,
    split_conflict,
)
from finrobot.engine.data.providers.fmp_provider import FMPProvider
from finrobot.engine.models.financial import (
    BalanceSheet,
    DataProvenance,
    FinancialData,
    HistoricalMetrics,
    IncomeStatement,
    MarketData,
    ValuationMetrics,
)

_QUARTERS = (
    date(2026, 3, 28),
    date(2025, 12, 27),
    date(2025, 9, 27),
    date(2025, 6, 28),
)
_CNY_TO_USD = 0.14  # synthetic spot, not a market fixing
_HKD_TO_USD = 0.128  # synthetic spot, distinct from the CNY rate on purpose


def _snapshot(**overrides: object) -> FinancialData:
    """A synthetic single-currency snapshot the publication gate can accept."""
    base = FinancialData(
        ticker="SYN",
        company_name="Synthetic Co",
        timestamp=datetime(2026, 4, 1, tzinfo=UTC),
        fiscal_period_end=_QUARTERS[0],
        ttm_quarter_ends=list(_QUARTERS),
        ttm_construction="four_quarters",
        income=IncomeStatement(
            revenue=400.0,
            ebitda=80.0,
            net_income=40.0,
            capital_expenditure=20.0,
        ),
        balance=BalanceSheet(total_debt=50.0, total_cash=10.0),
        market=MarketData(
            market_cap=1_000.0,
            shares_outstanding=10.0,
            current_price=100.0,
        ),
        valuation=ValuationMetrics(
            ebitda_operating=80.0,
            book_value_per_share=12.0,
        ),
        reporting_currency="USD",
        quote_currency="USD",
        provenance=DataProvenance(
            provider="synthetic",
            as_of=_QUARTERS[0],
            period_basis="ttm",
        ),
    )
    if overrides:
        return base.model_copy(update=overrides)
    return base


def _codes(reasons: list[str]) -> set[str]:
    return {item.split(":", 1)[0] for item in reasons}


class TestFxFieldPolicy:
    """Statement lines and the quote use their own rates, once."""

    def test_hkd_quote_cny_statements_do_not_share_a_rate(self) -> None:
        raw = _snapshot(
            ticker="0700.HK",
            share_class_scope="single_listing",
            reporting_currency="CNY",
            quote_currency="HKD",
        )
        raw.income.capital_expenditure = 70.0
        raw.valuation.ebitda_operating = 80.0
        converted = normalize_financialdata_to_usd(
            raw, _CNY_TO_USD, _HKD_TO_USD, rate_source="synthetic_spot"
        )
        assert converted.income.revenue == pytest.approx(400.0 * _CNY_TO_USD)
        assert converted.income.ebitda == pytest.approx(80.0 * _CNY_TO_USD)
        assert converted.income.capital_expenditure == pytest.approx(70.0 * _CNY_TO_USD)
        assert converted.valuation.ebitda_operating == pytest.approx(80.0 * _CNY_TO_USD)
        assert converted.valuation.book_value_per_share == pytest.approx(12.0 * _CNY_TO_USD)
        assert converted.market.current_price == pytest.approx(100.0 * _HKD_TO_USD)
        assert converted.market.market_cap == pytest.approx(1_000.0 * _HKD_TO_USD)
        assert converted.valuation.ebitda_operating == pytest.approx(converted.income.ebitda)
        assert converted.reporting_currency == "USD"
        assert converted.quote_currency == "USD"
        record = converted.fx_conversions[-1]
        assert record["scaler"] == "financialdata"
        assert record["rate_source"] == "synthetic_spot"
        assert "income.capital_expenditure" in record["scaled_fields"]
        assert "valuation.ebitda_operating" in record["scaled_fields"]
        assert "valuation.book_value_per_share" in record["scaled_fields"]
        assert "caller-supplied spot" in record["policy"]

    def test_cny_a_share_uses_one_rate_for_flows_and_quote(self) -> None:
        raw = _snapshot(
            ticker="300750.SZ",
            share_class_scope="single_listing",
            reporting_currency="CNY",
            quote_currency="CNY",
        )
        converted = normalize_financialdata_to_usd(
            raw, _CNY_TO_USD, _CNY_TO_USD, rate_source="synthetic_spot"
        )
        assert converted.income.capital_expenditure == pytest.approx(20.0 * _CNY_TO_USD)
        assert converted.valuation.ebitda_operating == pytest.approx(80.0 * _CNY_TO_USD)
        assert converted.market.current_price == pytest.approx(100.0 * _CNY_TO_USD)
        assert converted.valuation.ebitda_operating / converted.income.revenue == pytest.approx(
            80.0 / 400.0
        )

    def test_usd_quote_cny_adr_leaves_the_quote_unscaled(self) -> None:
        raw = _snapshot(ticker="BABA", reporting_currency="CNY", quote_currency="USD")
        converted = normalize_financialdata_to_usd(
            raw, _CNY_TO_USD, 1.0, rate_source="synthetic_spot"
        )
        assert converted.income.revenue == pytest.approx(400.0 * _CNY_TO_USD)
        assert converted.income.capital_expenditure == pytest.approx(20.0 * _CNY_TO_USD)
        assert converted.valuation.book_value_per_share == pytest.approx(12.0 * _CNY_TO_USD)
        assert converted.market.current_price == pytest.approx(100.0)
        assert converted.market.market_cap == pytest.approx(1_000.0)

    def test_pure_usd_is_not_converted(self) -> None:
        raw = _snapshot()
        again = normalize_financialdata_to_usd(raw, 999.0, 999.0)
        assert again is raw
        assert again.income.capital_expenditure == 20.0
        assert again.fx_conversions == []

    def test_second_conversion_does_not_scale_again(self) -> None:
        raw = _snapshot(reporting_currency="CNY", quote_currency="CNY")
        once = normalize_financialdata_to_usd(raw, _CNY_TO_USD, _CNY_TO_USD)
        twice = normalize_financialdata_to_usd(once, _CNY_TO_USD, _CNY_TO_USD)
        assert twice is once
        assert once.income.capital_expenditure == pytest.approx(20.0 * _CNY_TO_USD)
        assert len(once.fx_conversions) == 1


class TestTtmWindow:
    def test_four_nonoverlapping_quarters_are_accepted(self) -> None:
        window = select_statement_window(_QUARTERS)
        assert window.construction == "four_quarters"
        assert window.reasons == ()
        assert len(window.indexes) == 4

    def test_duplicate_quarter_is_not_summed(self) -> None:
        ends = (_QUARTERS[0], _QUARTERS[0], _QUARTERS[2], _QUARTERS[3])
        window = select_statement_window(ends)
        assert window.construction is None
        assert any(reason.startswith("ttm_overlap") for reason in window.reasons)

    def test_missing_quarter_is_not_summed(self) -> None:
        # 2026-03-28, 2025-12-27, then a jump to 2025-06-28: one quarter is absent.
        ends = (
            date(2026, 3, 28),
            date(2025, 12, 27),
            date(2025, 6, 28),
            date(2025, 3, 29),
        )
        window = select_statement_window(ends)
        assert window.construction is None
        assert any("ttm_window_unverified" in reason for reason in window.reasons)

    def test_two_halves_sum_only_the_latest_pair(self) -> None:
        ends = (
            date(2026, 6, 30),
            date(2025, 12, 31),
            date(2025, 6, 30),
            date(2024, 12, 31),
        )
        window = select_statement_window(ends)
        assert window.construction == "two_halves"
        assert window.indexes == (0, 1)

    def test_fy_plus_ytd_bridge_is_explicit(self) -> None:
        fy = FlowColumn(
            period_end=date(2024, 12, 31),
            kind="fy",
            currency="CNY",
            source="synthetic annual",
            values={"revenue": 1_000.0, "capital_expenditure": -100.0},
        )
        current = FlowColumn(
            period_end=date(2025, 9, 30),
            kind="ytd",
            currency="CNY",
            source="synthetic current YTD",
            values={"revenue": 800.0, "capital_expenditure": -90.0},
        )
        prior = FlowColumn(
            period_end=date(2024, 9, 30),
            kind="ytd",
            currency="CNY",
            source="synthetic prior YTD",
            values={"revenue": 700.0, "capital_expenditure": -60.0},
        )
        built = bridge_fy_ytd(fy, current, prior)
        assert built.status == "complete"
        assert built.construction == "fy_ytd_bridge"
        assert built.values["revenue"] == pytest.approx(1_100.0)
        assert built.capital_expenditure_outflow == pytest.approx(130.0)
        assert built.currency == "CNY"

    def test_untagged_columns_are_not_treated_as_ytd(self) -> None:
        quarterly = FlowColumn(
            period_end=date(2025, 9, 30),
            kind="quarter",
            currency="CNY",
            source="synthetic",
            values={"revenue": 10.0},
        )
        built = bridge_fy_ytd(quarterly, quarterly, quarterly)
        assert built.status == "incomplete"
        assert built.values == {}


class TestCapexSign:
    def test_negative_outflow_becomes_a_positive_magnitude(self) -> None:
        outflow, note = capex_outflow(-12.0)
        assert outflow == pytest.approx(12.0)
        assert note is None

    def test_positive_amount_is_not_flipped(self) -> None:
        outflow, note = capex_outflow(12.0)
        assert outflow is None
        assert note is not None and note.startswith("capex_sign_unverified")

    def test_fmp_refuses_a_positive_capex_sum(self) -> None:
        income = [
            {
                "date": day.isoformat(),
                "reportedCurrency": "USD",
                "revenue": 100.0,
                "operatingIncome": 20.0,
                "netIncome": 10.0,
                "depreciationAndAmortization": 1.0,
            }
            for day in _QUARTERS
        ]
        cash = [
            {"date": day.isoformat(), "capitalExpenditure": 5.0, "operatingCashFlow": 30.0}
            for day in _QUARTERS
        ]
        payload = FMPProvider._build_ttm_data(income, {}, {"currency": "USD"}, cash)
        assert payload["period_basis"] == "ttm"
        assert payload["capital_expenditure"] is None
        assert any("capex_sign_unverified" in reason for reason in payload["ttm_quality_reasons"])

    def test_fmp_does_not_sum_duplicate_quarters(self) -> None:
        income = [
            {"date": "2026-03-28", "reportedCurrency": "USD", "revenue": 100.0},
            {"date": "2026-03-28", "reportedCurrency": "USD", "revenue": 100.0},
            {"date": "2025-12-27", "reportedCurrency": "USD", "revenue": 100.0},
            {"date": "2025-09-27", "reportedCurrency": "USD", "revenue": 100.0},
        ]
        payload = FMPProvider._build_ttm_data(income, {}, {"currency": "USD"}, None)
        assert payload["period_basis"] == "unverified"
        assert payload["revenue"] is None
        assert payload["ttm_quarter_ends"] == []


class TestPublicationGate:
    def test_complete_synthetic_snapshot_has_no_block(self) -> None:
        assert publication_block_reasons(_snapshot()) == []

    def test_unknown_currency_blocks(self) -> None:
        reasons = publication_block_reasons(_snapshot(reporting_currency="UNKNOWN"))
        assert "currency_unspecified" in _codes(reasons)

    def test_missing_fx_blocks(self) -> None:
        snap = _snapshot()
        snap.provenance.degraded.append(DEGRADED_FX_UNAVAILABLE)
        reasons = publication_block_reasons(snap)
        assert "fx_missing" in _codes(reasons)

    def test_partial_fx_record_blocks(self) -> None:
        snap = _snapshot(
            fx_conversions=[
                {
                    "scaler": "financialdata",
                    "target": "USD",
                    "reporting_rate": _CNY_TO_USD,
                    "quote_rate": _HKD_TO_USD,
                    "scaled_fields": ["income.revenue", "income.ebitda"],
                }
            ]
        )
        reasons = publication_block_reasons(snap)
        assert any("income.capital_expenditure" in reason for reason in reasons)
        assert any("valuation.ebitda_operating" in reason for reason in reasons)
        assert any("valuation.book_value_per_share" in reason for reason in reasons)

    def test_unspecified_period_blocks(self) -> None:
        reasons = publication_block_reasons(_snapshot(provenance=None, ttm_quarter_ends=[]))
        assert "period_unspecified" in _codes(reasons)

    def test_exchange_suffix_without_share_scope_blocks(self) -> None:
        reasons = publication_block_reasons(_snapshot(ticker="0700.HK"))
        assert "share_class_scope_unverified" in _codes(reasons)

    def test_inconsistent_shares_block(self) -> None:
        reasons = publication_block_reasons(
            _snapshot(
                market=MarketData(
                    market_cap=1_000.0,
                    shares_outstanding=50.0,
                    current_price=100.0,
                )
            )
        )
        assert "shares_price_market_cap_inconsistent" in _codes(reasons)

    def test_split_that_does_not_match_price_blocks(self) -> None:
        reasons = publication_block_reasons(
            _snapshot(
                split_observation={
                    "old_shares": 10.0,
                    "new_shares": 20.0,
                    "old_price": 100.0,
                    "new_price": 100.0,
                }
            )
        )
        assert "split_unreconciled" in _codes(reasons)

    def test_split_with_inverse_price_does_not_block(self) -> None:
        assert split_conflict(10.0, 20.0, 100.0, 50.0) is None

    def test_ads_ratio_must_reproduce_market_cap(self) -> None:
        assert ads_quote_is_consistent(80.0, 8.0, 100.0, 1_000.0)
        assert not ads_quote_is_consistent(80.0, 1.0, 100.0, 1_000.0)


class TestEntityResolution:
    def test_private_name_is_not_mapped_to_a_nearby_ticker(self) -> None:
        # Synthetic catalog: a listed name that merely starts like the query.
        catalog = (("SPCX", "Space Acquisition Corp"),)
        resolved = resolve_listed_entity("SpaceX", catalog)
        assert resolved.status == "refused"
        assert resolved.ticker is None
        assert resolved.ticker != "SPCX"

    def test_exact_legal_name_resolves_once(self) -> None:
        catalog = (("NVDA", "NVIDIA Corporation"), ("BABA", "Alibaba Group Holding Limited"))
        resolved = resolve_listed_entity("NVIDIA Corporation", catalog)
        assert resolved.status == "unique_name"
        assert resolved.ticker == "NVDA"

    def test_ticker_not_in_the_catalog_is_not_rewritten(self) -> None:
        catalog = (("NVDA", "NVIDIA Corporation"),)
        resolved = resolve_listed_entity("BABA", catalog)
        assert resolved.status == "refused"
        assert resolved.ticker is None


@pytest.mark.asyncio
async def test_pipeline_refuses_formal_valuation_when_period_is_unknown() -> None:
    """A snapshot with no verified period does not produce a DCF or a recommendation."""
    from finrobot.engine.pipelines.equity_research import _execute_financial_modeling

    fd = _snapshot(provenance=None, ttm_quarter_ends=[], ttm_construction=None)
    historical = HistoricalMetrics(
        years=[],
        revenue=[],
        revenue_growth_yoy=[],
        cogs=[],
        gross_profit=[],
        gross_margin=[],
        sga=[],
        sga_ratio=[],
        ebitda=[],
        ebitda_margin=[],
        operating_income=[],
        operating_margin=[],
        net_income=[],
        eps=[],
        pe_ratio=[],
        cagr_revenue=None,
        ticker="SYN",
    )
    ctx: dict[str, object] = {"data_collection": fd, "historical_metrics": historical}
    deps = MagicMock()
    deps.settings.fmp_api_key = None
    out = await _execute_financial_modeling(MagicMock(), deps, "prompt", ctx, "SYN")
    assert out.structured is None
    assert "financial_modeling" not in ctx
    assert any("period_unspecified" in warning for warning in out.warnings)
    assert isinstance(ctx.get("input_quality_block"), list)
