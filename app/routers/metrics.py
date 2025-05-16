from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.schemas.metrics import MetricRequest
from app.services.arize.arize import ArizeClient

router = APIRouter(prefix="/metrics", tags=["Metrics"])


@router.post("")
async def metrics_endpoint(request: Request, metric_request: MetricRequest):
    try:
        getter_client: ArizeClient = request.app.state.arize_getter
        # start_time = metric_request.start_time
        session_id = metric_request.session_id
        agent_count, tool_count, token_usage = (
            getter_client.get_tracing_info_by_session_id(session_id)
        )
        response = {
            "agent_count": agent_count,
            "tool_count": tool_count,
            "token_usage": token_usage,
        }
        return JSONResponse(content=response, status_code=200)
    except Exception as e:
        error_response = {"error": str(e)}
        return JSONResponse(content=error_response, status_code=500)
