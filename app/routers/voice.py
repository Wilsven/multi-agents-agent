from typing import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from openinference.instrumentation import using_attributes

from app.schemas.voice import VoiceRequest
from app.services.openai import openai_agents_stream
from app.services.speech.text_to_speech import TextToSpeech

router = APIRouter(prefix="/voice", tags=["Voice"])


@router.post("/stream", response_class=StreamingResponse)
async def send_chat_stream(voice_request: VoiceRequest, request: Request):
    tts: TextToSpeech = request.app.state.text_to_speech_service

    print("Received voice request:", voice_request.message)

    async def response_generator() -> AsyncGenerator[bytes, None]:
        with using_attributes(session_id=voice_request.session_id):
            async for chunk in openai_agents_stream.main(
                request_type=voice_request.request_type,
                user_msg=voice_request.message,
                history=voice_request.history,
                current_agent=voice_request.agent_name,
                user_info=voice_request.user_info,
                speech_client=tts,
            ):

                # Convert ChatResponse to JSON bytes
                yield chunk.model_dump_json() + "\n"

    return StreamingResponse(response_generator(), media_type="application/x-ndjson")
