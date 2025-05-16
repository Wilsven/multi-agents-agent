from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from openinference.instrumentation import using_attributes

from app.schemas.chat import ChatRequest
from app.services.openai import (
    openai_agents_stream,
    openai_agents_stream_mcp,
)

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/stream", response_class=StreamingResponse)
async def send_chat_stream(chat_request: ChatRequest):
    print("Received chat request:", chat_request.message)

    async def response_generator() -> AsyncGenerator[bytes, None]:
        with using_attributes(session_id=chat_request.session_id):
            async for chunk in openai_agents_stream.main(
                request_type=chat_request.request_type,
                user_msg=chat_request.message,
                history=chat_request.history,
                current_agent=chat_request.agent_name,
                auth_token=chat_request.auth_token,
                speech_client=None,
            ):
                # Convert ChatResponse to JSON bytes
                yield chunk.model_dump_json() + "\n"

    return StreamingResponse(response_generator(), media_type="application/x-ndjson")


@router.post("/stream_mcp", response_class=StreamingResponse)
async def send_chat_stream_general(chat_request: ChatRequest):
    print("Received chat request:", chat_request.message)

    async def response_generator() -> AsyncGenerator[bytes, None]:
        with using_attributes(session_id=chat_request.session_id):
            async for chunk in openai_agents_stream_mcp.main_mcp(
                request_type=chat_request.request_type,
                user_msg=chat_request.message,
                history=chat_request.history,
                current_agent=chat_request.agent_name,
                auth_token=chat_request.auth_token,
                speech_client=None,
            ):
                # Convert ChatResponse to JSON bytes
                yield chunk.model_dump_json() + "\n"

    return StreamingResponse(response_generator(), media_type="application/x-ndjson")
