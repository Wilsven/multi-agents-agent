from typing import Optional

from pydantic import BaseModel

from app.schemas.chat import RequestBase, RequestType, ResponseBase


class TranscriptionResponse(BaseModel):
    text: str


class VoiceRequest(RequestBase):
    request_type: RequestType = RequestType.VOICE_REQUEST


class VoiceResponse(ResponseBase):
    audio_data: Optional[str] = None
