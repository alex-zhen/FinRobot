"""Official capex construction stays behind the publication gate.

Figures below are either synthetic or recomputed from the vendored evidence
files' own component rows. A collection flag is not a valuation pass.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from finrobot.config import deepseek_api_key_from_process, sec_user_agent_from_process
from finrobot.engine.data.interface import DataResult, ProviderError
from finrobot.engine.data.layer import DataLayer
from finrobot.engine.data.normalize.valuation_input import publication_block_reasons
from finrobot.engine.data.types import DataType
from finrobot.engine.models.financial import (
    DataProvenance,
    FinancialData,
    IncomeStatement,
    MarketData,
    ThesisResult,
    ValuationMetrics,
)
from finrobot.runners.equity_batch import (
    apply_official_ttm_capex,
    gate_reasons,
    require_key_for_open_gate,
    resolved_sec_user_agent,
    settings_for_box,
)
from finrobot.runners.evidence_loader import baba_line_records, tencent_same_day_share_record
from finrobot.runners.evidence_loader import (
    load_finrobot_evidence_capex,
    load_tencent_period_end_shares,
    load_tencent_statutory_capex,
)
from finrobot.runners.official_inputs import (
    OfficialAmount,
    construct_official_ttm_capex,
    label_self_calculated_ocf_minus_capex,
    period_end_ordinary_share_block,
)
from finrobot.runners.serialize import dumps_model, loads_model

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "official"
_QUARTERS = (
    date(2026, 3, 28),
    date(2025, 12, 27),
    date(2025, 9, 27),
    date(2025, 6, 28),
)
_URL = "https://example.test/filing"


def _amount(
    *,
    ticker: str = "SYN",
    role: str = "statutory_capex",
    value: float = -1.0,
    period_end: date = date(2026, 3, 31),
    period_type: str = "quarter",
    currency: str = "USD",
) -> OfficialAmount:
    return OfficialAmount(
        ticker=ticker,
        source_url=_URL,
        source_kind="filing",
        statement_line="Purchases of property and equipment",
        line_item="capital_expenditure",
        line_role=role,  # type: ignore[arg-type]
        period_start=period_end,
        period_end=period_end,
        period_type=period_type,  # type: ignore[arg-type]
        currency=currency,
        unit="million",
        value=value,
    )


def _snapshot(**overrides: object) -> FinancialData:
    base = FinancialData(
        ticker="SYN",
        company_name="Synthetic Co",
        timestamp=datetime(2026, 4, 1, tzinfo=UTC),
        fiscal_period_end=_QUARTERS[0],
        ttm_quarter_ends=list(_QUARTERS),
        ttm_construction="four_quarters",
        income=IncomeStatement(
            revenue=400.0, ebitda=80.0, net_income=40.0, capital_expenditure=20.0
        ),
        market=MarketData(market_cap=1_000.0, shares_outstanding=10.0, current_price=100.0),
        valuation=ValuationMetrics(),
        reporting_currency="USD",
        quote_currency="USD",
        provenance=DataProvenance(provider="synthetic", as_of=_QUARTERS[0], period_basis="ttm"),
    )
    if overrides:
        return base.model_copy(update=overrides)
    return base


def _codes(reasons: list[str]) -> set[str]:
    return {item.split(":", 1)[0] for item in reasons}


def test_four_quarters_ignore_management_and_company_fcf() -> None:
    statutory = [_amount(value=-(index + 1), period_end=end) for index, end in enumerate(_QUARTERS)]
    extras = [
        _amount(role="management_capex", value=-999.0),
        _amount(role="company_fcf", value=-50.0),
        _amount(role="fcf_recon_ppe", value=-80.0),
        _amount(role="self_calculated_fcf", value=-70.0),
    ]
    built = construct_official_ttm_capex([*statutory, *extras], "SYN")
    assert built.ok
    assert built.construction == "four_quarters"
    assert built.outflow == pytest.approx(10_000_000.0)
    assert built.period_ends == _QUARTERS


def test_full_year_capex_is_not_ttm() -> None:
    built = construct_official_ttm_capex(
        [_amount(ticker="BABA", value=-126_063.0, period_end=date(2026, 3, 31), period_type="fy")],
        "BABA",
    )
    assert built.outflow is None
    assert built.reason is not None
    assert "not a trailing twelve months" in built.reason


def test_self_calculated_fcf_is_not_a_company_definition() -> None:
    labeled = label_self_calculated_ocf_minus_capex(102_718.0, 6_042.0)
    assert labeled["company_defined"] is False
    assert labeled["label"] == "self_calculated_ocf_minus_statutory_capex"
    assert labeled["value"] == pytest.approx(96_676.0)


def test_tencent_bridge_uses_statutory_line_only() -> None:
    path = _FIXTURES / "tencent_valuation_evidence.json"
    rows = load_tencent_statutory_capex(path)
    assert len(rows) == 3
    assert {row.line_role for row in rows} == {"statutory_capex"}
    built = construct_official_ttm_capex(rows, "0700.HK")
    assert built.ok
    assert built.construction == "fy_ytd_bridge"
    assert built.outflow == pytest.approx(137_539 * 1_000_000)
    shares, as_of = load_tencent_period_end_shares(path)
    assert shares == 9_092_234_841
    assert as_of == date(2026, 6, 30)
    document = json.loads(path.read_text(encoding="utf-8"))
    waso = document["periods"]["H1_2026"]["shares"]["basic_waso_million"]["amount"]
    assert shares != waso


def test_baba_fy_and_narrow_q1_do_not_make_ttm_capex() -> None:
    path = _FIXTURES / "finrobot_input_evidence.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    statutory = next(
        field
        for period in document["issuers"]["Alibaba_Group_Holding_Limited"]["periods"].values()
        for field in period["fields"]
        if field["field_id"] == "BABA.FY2026.capex_acquisitions_of_land_ppe"
    )
    assert statutory["usable_as_finrobot_input"] is True
    rows = load_finrobot_evidence_capex(path)
    baba = [row for row in rows if row.ticker == "BABA"]
    assert {row.line_role for row in baba} == {"statutory_capex", "fcf_recon_ppe", "company_fcf"}
    statutory_rows = [row for row in baba if row.line_role == "statutory_capex"]
    assert len(statutory_rows) == 1
    assert statutory_rows[0].period_type == "fy"
    assert statutory_rows[0].value == -126_063
    assert all(
        row.value != statutory_rows[0].value for row in baba if row.line_role != "statutory_capex"
    )
    assert all(row.value != -49_850 for row in baba)
    built = construct_official_ttm_capex(rows, "BABA")
    assert built.outflow is None
    assert built.reason is not None
    assert "not a trailing twelve months" in built.reason


def test_nvda_official_ytd_without_prior_ytd_is_not_ttm() -> None:
    path = _FIXTURES / "finrobot_input_evidence.json"
    rows = load_finrobot_evidence_capex(path)
    nvda = [row for row in rows if row.ticker == "NVDA"]
    assert {row.line_role for row in nvda} == {"statutory_capex"}
    assert {row.period_type for row in nvda} == {"fy", "ytd"}
    built = construct_official_ttm_capex(rows, "NVDA")
    assert built.outflow is None
    assert built.reason is not None
    assert "not a trailing twelve months" in built.reason


def test_period_end_shares_require_the_same_day_price() -> None:
    as_of = date(2026, 6, 30)
    assert (
        period_end_ordinary_share_block(
            shares=100.0,
            share_as_of=as_of,
            price=10.0,
            price_as_of=as_of,
            market_cap=1_000.0,
        )
        is None
    )
    mismatch = period_end_ordinary_share_block(
        shares=9_092_234_841,
        share_as_of=as_of,
        price=400.0,
        price_as_of=date(2026, 11, 16),
        market_cap=3_600_000_000_000,
    )
    assert mismatch is not None
    assert mismatch.startswith("share_price_date_unmatched")


def test_applied_official_capex_still_faces_the_publication_gate() -> None:
    built = construct_official_ttm_capex(
        load_tencent_statutory_capex(_FIXTURES / "tencent_valuation_evidence.json"),
        "0700.HK",
    )
    usd = _snapshot(capital_expenditure=None)
    # model_copy update replaces the income object only when passed as income.
    usd.income.capital_expenditure = None
    untouched, refused = apply_official_ttm_capex(usd, built)
    assert refused is not None
    assert untouched.income.capital_expenditure is None

    cny = _snapshot(
        ticker="0700.HK",
        reporting_currency="CNY",
        quote_currency="HKD",
        ttm_construction=None,
        ttm_quarter_ends=[],
        fiscal_period_end=None,
    )
    cny.income.capital_expenditure = None
    cny.provenance = DataProvenance(
        provider="synthetic",
        as_of=None,
        period_basis="unverified",
        degraded=["period_basis_unknown"],
    )
    updated, applied = apply_official_ttm_capex(cny, built)
    assert applied is None
    assert updated.income.capital_expenditure == pytest.approx(137_539 * 1_000_000)
    assert updated.ttm_construction == "fy_ytd_bridge"
    assert updated.provenance is not None
    assert updated.provenance.period_basis == "ttm"
    assert "period_basis_unknown" not in updated.provenance.degraded
    reasons = gate_reasons(
        updated,
        official_shares=9_092_234_841,
        share_as_of=date(2026, 6, 30),
        price_as_of=date(2026, 11, 16),
    )
    assert "currency_not_unified" in _codes(reasons)
    assert "share_price_date_unmatched" in _codes(reasons)
    assert publication_block_reasons(updated)


def test_thesis_result_roundtrip_is_not_a_dict_dump() -> None:
    thesis = ThesisResult(
        recommendation="Buy",
        price_target=10.0,
        price_target_basis="DCF",
        catalysts=["synthetic driver"],
        risks=["synthetic risk"],
        narrative="synthetic narrative",
    )
    with pytest.raises(TypeError):
        json.dumps(thesis)
    restored = loads_model(dumps_model(thesis))
    assert isinstance(restored, ThesisResult)
    assert restored.recommendation == "Buy"
    assert restored.price_target == 10.0


def test_box_settings_use_the_injected_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert deepseek_api_key_from_process() is None
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        require_key_for_open_gate()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "box-process-key")
    assert deepseek_api_key_from_process() == "box-process-key"
    settings = settings_for_box("box-process-key")
    assert settings.model_name == "deepseek:deepseek-chat"
    assert settings.language == "zh"
    assert settings.provider_key("deepseek") == "box-process-key"
    provider = next(item for item in settings.custom_providers if item.id == "deepseek")
    assert provider.base_url == "https://api.deepseek.com"


def test_placeholder_sec_identity_is_not_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FINROBOT_SEC_USER_AGENT", raising=False)
    assert sec_user_agent_from_process() is None
    assert resolved_sec_user_agent() is None
    monkeypatch.setenv("FINROBOT_SEC_USER_AGENT", "FinRobot admin@example.com")
    assert sec_user_agent_from_process() is None
    settings = settings_for_box("box-process-key")
    assert settings.sec_user_agent == "FinRobot admin@example.com"
    monkeypatch.setenv("FINROBOT_SEC_USER_AGENT", "Ada Lovelace ada@research.test")
    assert resolved_sec_user_agent() == "Ada Lovelace ada@research.test"
    configured = settings_for_box("box-process-key")
    assert configured.sec_user_agent == "Ada Lovelace ada@research.test"


def test_baba_statutory_ttm_stays_blocked_while_alternatives_are_labeled() -> None:
    record = baba_line_records(_FIXTURES / "baba_statutory_capex_ttm.json")
    assert record["statutory_ttm_capex"] is None
    assert record["statutory_ttm_blocked"] is True
    assert record["management_capex_ttm"] is None
    dumped = json.dumps(record)
    assert "155065" not in dumped
    narrow = record["narrow_ppe_alternative"]
    assert narrow["opens_publication_gate"] is False
    assert narrow["signed_value"] == pytest.approx(-151_052)
    ocf = record["ocf_ttm"]
    assert ocf["opens_publication_gate"] is False
    assert ocf["signed_value"] == pytest.approx(78_486)
    assert ocf["self_constructed"] is True
    baseline = record["historical_fy_baseline"]
    assert baseline["usable_as_latest_ttm"] is False
    assert baseline["usable_as_current_analysis"] is False
    assert baseline["statutory_capex_rmb_million"] == -126_063
    assert baseline["period_end"] == "2026-03-31"


def test_tencent_same_day_product_is_not_an_independent_or_current_market_cap() -> None:
    historical = tencent_same_day_share_record(
        _FIXTURES / "tencent_20260630_price_mcap.json",
        valuation_as_of=date(2026, 6, 30),
    )
    assert historical["independent_market_cap"] is False
    assert historical["price_is_official_exchange_file"] is False
    assert historical["third_party_market_cap_used"] is False
    assert historical["share_class_scope_applied_to_live_snapshot"] is False
    assert historical["value"] == pytest.approx(9_092_234_841 * 429.80)
    assert historical["usable_as_current_price"] is True
    assert historical["historical_share_gate"] is None
    assert historical["value"] != pytest.approx(3.87e12)
    later = tencent_same_day_share_record(
        _FIXTURES / "tencent_20260630_price_mcap.json",
        valuation_as_of=date(2026, 11, 17),
    )
    assert later["usable_as_current_price"] is False
    assert later["historical_share_gate"] is None
    assert later["value"] == pytest.approx(9_092_234_841 * 429.80)
    assert (
        period_end_ordinary_share_block(
            shares=9_092_234_841,
            share_as_of=date(2026, 6, 30),
            price=429.80,
            price_as_of=date(2026, 6, 30),
            market_cap=None,
        )
        is not None
    )
    assert (
        period_end_ordinary_share_block(
            shares=9_092_234_841,
            share_as_of=date(2026, 6, 30),
            price=429.80,
            price_as_of=date(2026, 6, 30),
            market_cap=3.87e12,
            market_cap_basis="self_calculated_price_times_issued_shares",
        )
        is not None
    )
    mismatch = period_end_ordinary_share_block(
        shares=9_092_234_841,
        share_as_of=date(2026, 6, 30),
        price=400.0,
        price_as_of=date(2026, 11, 17),
        market_cap=3.87e12,
    )
    assert mismatch is not None
    assert mismatch.startswith("share_price_date_unmatched")


async def test_news_failures_do_not_open_the_price_circuit() -> None:
    provider = MagicMock()
    provider.name = "yfinance"
    provider.capabilities.return_value = [DataType.NEWS, DataType.PRICE]

    async def fetch(ticker: str, data_type: DataType, **kwargs: object) -> DataResult:
        if data_type == DataType.NEWS:
            raise ProviderError("yahoo news down")
        return DataResult(
            data={"current_price": 100.0},
            provider="yfinance",
            ticker=ticker,
            data_type=DataType.PRICE,
            timestamp=datetime(2026, 11, 17, tzinfo=timezone.utc),
        )

    provider.fetch = fetch
    cache = MagicMock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock()
    layer = DataLayer(providers=[provider], cache=cache)
    for _ in range(3):
        await layer.fetch(DataType.NEWS, "NVDA")
    assert layer._health.is_available("yfinance")
    price = await layer.fetch(DataType.PRICE, "NVDA")
    assert price.data["current_price"] == 100.0
