"""Business, financial, and risk notes when a verified valuation is refused.

This is not a valuation model. It does not open the FCFF gate, does not treat
a share as-of as the quote date, and cannot publish a point target or a
buy/hold/sell recommendation.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

_INSTRUCTIONS = (
    "Write a business, financial, and risk note from the supplied evidence only. "
    "Do not invent figures, peers, or a valuation. Do not give a buy, hold, or sell "
    "recommendation and do not give a target price. The JSON facts are deterministic. "
    "If capex_present is true, do not say cash flow or statutory capex was missing; "
    "state the refusal codes that are actually present. If statutory capex is the "
    "refusal, say the FCFF DCF was not run. Company-defined free cash flow is a "
    "liquidity risk, not FCFF. If a share count and a price have different dates, "
    "state both dates and do not call them the same day. business_background lists "
    "segment names only: they are not officially named peers, not a multiple average, "
    "and not semiconductor roles. This note is not a complete analysis."
)


class UnvaluedResearch(BaseModel):
    """Narrative research with the valuation fields forced closed."""

    business_observations: str = Field(min_length=40)
    financial_observations: str = Field(min_length=40)
    risks: list[str] = Field(min_length=1)
    recommendation: Literal["withheld"] = "withheld"
    price_target: None = None
    dcf_available: Literal[False] = False

    @model_validator(mode="before")
    @classmethod
    def _force_withhold(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            data["recommendation"] = "withheld"
            data["price_target"] = None
            data["dcf_available"] = False
        return data


def deterministic_facts(
    *,
    inputs: dict[str, Any] | None,
    reasons: list[str],
    official_flows: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Facts the model is not allowed to contradict."""
    present = bool(inputs and inputs.get("capex_present"))
    codes = [item.split(":", 1)[0] for item in reasons]
    capex_is_refusal = "capex_missing" in codes or "capex_sign_unverified" in codes
    return {
        "capex_present": present or bool(official_flows and official_flows.get("capex_present")),
        "capex_is_the_refusal": capex_is_refusal,
        "refusal_codes": codes,
        "ttm_construction": None if not inputs else inputs.get("ttm_construction"),
        "period_basis": None if not inputs else inputs.get("period_basis"),
        "official_flows": official_flows,
        "share_as_of_is_not_the_quote_date": "share_price_date_unmatched" in codes,
        "share_class_scope_unverified": "share_class_scope_unverified" in codes,
    }


def facts_text(facts: dict[str, Any]) -> str:
    """One paragraph a model opinion is not allowed to replace."""
    flows = facts.get("official_flows") if isinstance(facts.get("official_flows"), dict) else None
    parts: list[str] = []
    if facts.get("capex_present"):
        if flows:
            parts.append(
                "Official TTM cash flows are present for "
                f"{flows.get('window')} ({flows.get('construction')}, "
                f"{flows.get('currency')} {flows.get('unit')}): "
                f"revenue {flows.get('revenue')}, "
                f"operating cash flow {flows.get('operating_cash_flow')}, "
                f"statutory capex outflow {flows.get('statutory_capex_outflow')} "
                f"on the line {flows.get('capex_line')}."
            )
        else:
            parts.append("Statutory capex is present on the supplied snapshot.")
        if not facts.get("capex_is_the_refusal"):
            parts.append("Missing capex is not the reason verified valuation was refused.")
    if facts.get("capex_is_the_refusal"):
        parts.append("Statutory capex is why FCFF was not run.")
    codes = facts.get("refusal_codes") or []
    if codes:
        parts.append("Refusal codes: " + ", ".join(str(item) for item in codes) + ".")
    if facts.get("share_as_of_is_not_the_quote_date"):
        parts.append("The share-count date and the price date are not the same day.")
    if facts.get("share_class_scope_unverified"):
        parts.append("Share-class scope is unverified.")
    background = facts.get("business_background")
    if isinstance(background, dict) and background.get("issuers"):
        names = []
        for issuer in background["issuers"]:
            if not isinstance(issuer, dict):
                continue
            label = issuer.get("name") or "issuer"
            segments = ", ".join(str(item) for item in issuer.get("segment_names") or [])
            names.append(f"{label}: {segments}" if segments else str(label))
        parts.append(
            "Business background only, not officially named peers and not a multiple "
            "average: " + "; ".join(names) + "."
        )
    parts.append("This is not a complete analysis and it has no point target.")
    return " ".join(parts)


def _denies_supplied_flows(text: str) -> bool:
    lowered = text.lower()
    if "no income statement" in lowered or "no cash-flow" in lowered or "no cash flow" in lowered:
        return True
    if "no statutory capex" in lowered:
        return True
    if ("没有" in text or "未提供" in text or "缺失" in text) and (
        "现金流" in text or "资本开支" in text
    ):
        return True
    return False


def _denies_supplied_business(text: str, facts: dict[str, Any]) -> bool:
    background = facts.get("business_background")
    if not isinstance(background, dict) or not background.get("issuers"):
        return False
    lowered = text.lower()
    return (
        "no business" in lowered
        or "no segment" in lowered
        or "no description of operations" in lowered
    )


def compose_unvalued_publication(
    model: dict[str, Any] | None, facts: dict[str, Any]
) -> dict[str, Any]:
    """Lead with deterministic facts. A model denial does not erase them."""
    published = facts_text(facts)
    opinion = None if model is None else dict(model)
    discarded: list[str] = []
    if opinion is not None and facts.get("capex_present"):
        financial = opinion.get("financial_observations")
        if isinstance(financial, str) and _denies_supplied_flows(financial):
            discarded.append(financial)
            opinion["financial_observations"] = published
        risks = opinion.get("risks")
        if isinstance(risks, list):
            kept = []
            for item in risks:
                if isinstance(item, str) and _denies_supplied_flows(item):
                    discarded.append(item)
                else:
                    kept.append(item)
            opinion["risks"] = kept
        business = opinion.get("business_observations")
        if isinstance(business, str) and _denies_supplied_business(business, facts):
            discarded.append(business)
            opinion["business_observations"] = (
                "Segment names in the facts are business background only. "
                "They are not officially named peers and this is not a complete analysis."
            )
        opinion["recommendation"] = "withheld"
        opinion["price_target"] = None
        opinion["dcf_available"] = False
    return {
        "facts_text": published,
        "deterministic_facts": facts,
        "model_opinion": opinion,
        "discarded_claims": discarded,
        "dcf_available": False,
        "point_target": None,
        "recommendation": "withheld",
        "complete_analysis": False,
    }


def unvalued_research_allowed(disposition: str) -> bool:
    """A refused verified valuation can still have a non-valuation note."""
    return disposition in {"dcf_refused_other_research_ok", "refused"}


async def run_unvalued_research(
    ticker: str, api_key: str, evidence: dict[str, Any]
) -> UnvaluedResearch:
    """Call the configured chat model and force the valuation fields closed."""
    from pydantic_ai import Agent

    from finrobot.runners.equity_batch import settings_for_box

    settings = settings_for_box(api_key)
    agent = Agent(
        settings.create_model(),
        output_type=UnvaluedResearch,
        instructions=_INSTRUCTIONS,
        defer_model_check=True,
    )
    result = await agent.run(
        f"Ticker {ticker}. Evidence JSON follows. Use only these facts.\n{evidence}"
    )
    return result.output
