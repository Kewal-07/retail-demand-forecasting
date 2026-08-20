"""POST /explain. See CLAUDE.md Section 9/10.

prediction_id is always the id of whichever /predict response is
currently rendered, so an answer reflects the active scenario, not the
page's original default -- each prediction_id maps to its own stored
context, built at /predict time from whatever scenario was active then.
"""
from fastapi import APIRouter, HTTPException, Request

from api.schemas import ExplainRequest, ExplainResponse
from src.explain.generate import call_groq

router = APIRouter()


def _grounded_on(context: dict) -> list[str]:
    return [key for key, value in context.items() if value]


@router.post("/explain", response_model=ExplainResponse)
def explain(body: ExplainRequest, request: Request) -> ExplainResponse:
    store = request.app.state.prediction_store
    if body.prediction_id not in store:
        raise HTTPException(
            status_code=404, detail=f"prediction_id {body.prediction_id!r} not found"
        )

    context = store[body.prediction_id]["context"]
    explanation = call_groq(context, question=body.question, history=body.history)

    return ExplainResponse(explanation=explanation, grounded_on=_grounded_on(context))
