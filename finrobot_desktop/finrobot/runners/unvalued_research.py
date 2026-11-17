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
    "recommendation and do not give a target price. If statutory capex is missing, "
    "say the FCFF DCF was not run. Company-defined free cash flow is a liquidity "
    "risk, not FCFF. If a share count and a price have different dates, state both "
    "dates and do not call them the same day. Semiconductor role labels do not apply "
    "to companies outside that industry."
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
