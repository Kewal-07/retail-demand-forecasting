"""Explanation generation. See CLAUDE.md Section 6E and Section 9.

Deterministic pipeline, no agent loop: assemble context from five
sources, format into a prompt, generate via Groq. If Groq is
unavailable (no key, rate limited, network error), fall back to a
template built from SHAP values alone rather than failing the request
-- see Section 10's failure handling table ("Groq rate limited ->
templated explanation from SHAP only").
"""
import os

from groq import Groq

MODEL = "allam-2-7b"
# Groq's model lineup changed substantially since this constant was first
# picked (llama-3.1-8b-instant was deprecated by the time we tested with
# a real key). allam-2-7b is a plain instruct model -- picked deliberately
# over Groq's reasoning models (gpt-oss, qwen3.6, which spend part of the
# token budget on hidden <think> reasoning and risk truncated/empty
# output at MAX_TOKENS=200) and over its agentic "compound" models (which
# can call tools/search internally, conflicting with the "no agent loop"
# design principle above). See DEVLOG.md for the comparison.
TEMPERATURE = 0.2
MAX_TOKENS = 200
MAX_DRIVERS_NAMED = 3

SYSTEM_PROMPT = (
    "You are explaining a retail demand forecast to an operations audience. "
    "Ground every claim in the provided context only. Name at most "
    f"{MAX_DRIVERS_NAMED} drivers, using ONLY the definitions given for them -- "
    "never invent or guess what a feature name means. Never make causal claims "
    "-- describe correlations and model attributions, not causes. Be concise and concrete."
)

# FEATURE_COLUMNS is a fixed, known 40-name vocabulary for this project (see
# src/features/pipeline.py), so a glossary is small and complete rather than
# open-ended. Without this, a real test call named "snap_active" as a driver
# and the model confidently explained it as "the number of Snapchat users
# actively using the platform" -- a plausible-sounding but completely wrong
# guess. snap_active means something specific and non-obvious (SNAP benefit
# activation) that no model should be trusted to infer from the name alone.
FEATURE_GLOSSARY = {
    "cat_id": "product category",
    "day_of_month": "day of the month",
    "days_since_last_event": "days since the last calendar event (holiday/promo)",
    "days_since_last_sale": "days since this item last recorded any sales",
    "days_to_next_event": "days until the next calendar event (holiday/promo)",
    "dept_id": "product department",
    "event_cat": "calendar event type crossed with product category",
    "event_name_1": "name of the primary calendar event on this date",
    "event_name_2": "name of a secondary calendar event on this date, if any",
    "event_type_1": "type of the primary calendar event (e.g. religious, sporting, cultural, national)",
    "event_type_2": "type of a secondary calendar event, if any",
    "is_closed": "whether the store is closed this day (Christmas only)",
    "item_id": "product identifier",
    "lag_28": "this item's own sales exactly 28 days earlier",
    "lag_35": "this item's own sales exactly 35 days earlier",
    "lag_42": "this item's own sales exactly 42 days earlier",
    "lag_56": "this item's own sales exactly 56 days earlier",
    "lag_364": "this item's own sales exactly 364 days earlier (same day, one year prior)",
    "mean_nonzero_sales": "this item's historical average sales on days it sold at all",
    "month": "calendar month",
    "price_change_1w": "how much this item's price changed over the past week",
    "price_rank_dept": "this item's price percentile within its department that week",
    "price_rel_mean": "this item's current price relative to its own historical average price",
    "sell_price": "this item's current price",
    "snap_CA": "whether SNAP (food assistance program) benefits are active in California this day",
    "snap_TX": "whether SNAP benefits are active in Texas this day",
    "snap_WI": "whether SNAP benefits are active in Wisconsin this day",
    "snap_active": (
        "whether SNAP (Supplemental Nutrition Assistance Program) benefits are "
        "active in this item's own state this day -- NOT related to the Snapchat app"
    ),
    "state_id": "US state",
    "store_id": "store identifier",
    "roll_mean_7_28": "this item's average sales over a trailing 7-day window, as of 28 days ago",
    "roll_mean_28_28": "this item's average sales over a trailing 28-day window, as of 28 days ago",
    "roll_mean_56_28": "this item's average sales over a trailing 56-day window, as of 28 days ago",
    "roll_std_7_28": "volatility (day-to-day std dev) of this item's sales over a trailing 7-day window, as of 28 days ago",
    "roll_std_28_28": "volatility of this item's sales over a trailing 28-day window, as of 28 days ago",
    "roll_std_56_28": "volatility of this item's sales over a trailing 56-day window, as of 28 days ago",
    "wday": "day of the week",
    "week_of_year": "calendar week of the year",
    "weeks_since_price_change": "weeks since this item's price last changed",
    "zero_share_28": "share of the last 28 days this item had zero sales (a measure of intermittency)",
}


def build_explanation_context(
    forecast: dict,
    shap_top5: list,
    calendar_events,
    price_history,
    eda_passages: list[str],
) -> dict:
    return {
        "forecast": forecast,
        "shap_top5": shap_top5,
        "calendar_events": calendar_events,
        "price_history": price_history,
        "eda_passages": eda_passages,
    }


def _format_context_prompt(context: dict, question: str | None) -> str:
    top_drivers = context["shap_top5"][:MAX_DRIVERS_NAMED]
    driver_definitions = "\n".join(
        f"- {d}: {FEATURE_GLOSSARY.get(d, 'an internal model feature; do not guess its meaning')}"
        for d in top_drivers
    )
    lines = [
        f"Forecast: {context['forecast']}",
        f"Top drivers, with their exact definitions (use ONLY these, never guess):\n{driver_definitions}",
        f"Calendar events in window: {context['calendar_events']}",
        f"Recent price history: {context['price_history']}",
        f"Related EDA notes: {context['eda_passages']}",
    ]
    lines.append(f"Question: {question}" if question else "Write a short paragraph explaining this forecast.")
    return "\n\n".join(lines)


def _template_fallback(context: dict) -> str:
    top_drivers = context["shap_top5"][:MAX_DRIVERS_NAMED]
    p50 = (context.get("forecast") or {}).get("p50")
    total_str = f" (total forecast demand: {sum(p50):.1f} units over the window)" if p50 else ""

    if not top_drivers:
        return f"No SHAP attribution is available for this forecast{total_str}."
    names = ", ".join(str(d) for d in top_drivers)
    return f"This forecast{total_str} is primarily driven by {names}, based on the model's own attributions."


def call_groq(context: dict, question: str | None, history: list[dict]) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return _template_fallback(context)

    try:
        client = Groq(api_key=api_key)
        prompt = _format_context_prompt(context, question)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": prompt},
        ]
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return response.choices[0].message.content
    except Exception:
        return _template_fallback(context)
