from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.services.speech.speech_to_text import SpeechToText

router = APIRouter(prefix="/transcription", tags=["Transcription"])


@router.post("")
async def transcribe_endpoint(request: Request):
    form = await request.form()
    audio_file = form.get("file")

    if not audio_file:
        return JSONResponse(content={"error": "No file provided"}, status_code=400)

    stt: SpeechToText = request.app.state.speech_to_text_service
    response, status_code = await stt.transcribe(audio_file)
    return JSONResponse(content=response, status_code=status_code)
