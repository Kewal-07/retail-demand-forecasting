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

MODEL = "llama-3.1-8b-instant"
TEMPERATURE = 0.2
MAX_TOKENS = 200
MAX_DRIVERS_NAMED = 3

SYSTEM_PROMPT = (
    "You are explaining a retail demand forecast to an operations audience. "
    "Ground every claim in the provided context only. Name at most "
    f"{MAX_DRIVERS_NAMED} drivers. Never make causal claims -- describe "
    "correlations and model attributions, not causes. Be concise and concrete."
)


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
    lines = [
        f"Forecast: {context['forecast']}",
        f"Top drivers: {top_drivers}",
        f"Calendar events in window: {context['calendar_events']}",
        f"Recent price history: {context['price_history']}",
        f"Related EDA notes: {context['eda_passages']}",
    ]
    lines.append(f"Question: {question}" if question else "Write a short paragraph explaining this forecast.")
    return "\n\n".join(lines)


def _template_fallback(context: dict) -> str:
    top_drivers = context["shap_top5"][:MAX_DRIVERS_NAMED]
    if not top_drivers:
        return "No SHAP attribution is available for this forecast."
    names = ", ".join(str(d) for d in top_drivers)
    return f"This forecast is primarily driven by {names}, based on the model's own attributions."


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
