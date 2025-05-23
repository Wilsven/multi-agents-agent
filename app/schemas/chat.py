from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Optional

from pydantic import BaseModel


class RequestType(StrEnum):
    CHAT_REQUEST = "chat_request"
    VOICE_REQUEST = "voice_request"


class RequestBase(BaseModel):
    request_type: RequestType
    message: str
    history: Optional[list] = None
    agent_name: Optional[str] = None
    auth_token: str
    session_id: str = None


class ChatRequest(RequestBase):
    request_type: RequestType = RequestType.CHAT_REQUEST


class EventType(StrEnum):
    DELTA_TEXT_EVENT = "delta_text_event"
    COMPLETED_TEXT_EVENT = "completed_text_event"
    NEW_AGENT_EVENT = "new_agent_event"
    TOOL_CALL_EVENT = "tool_call_event"
    TOOL_CALL_OUTPUT_EVENT = "tool_call_output_event"
    TERMINATING_EVENT = "terminating_event"


class DataType(StrEnum):
    BOOKING_DETAILS = "booking_details"
    RESCHEDULE_DETAILS = "reschedule_details"
    CANCEL_DETAILS = "cancel_details"


class ResponseBase(BaseModel):
    agent_name: str
    history: Optional[list] = None
    event_type: EventType
    message: Optional[str] = None
    delta_message: Optional[str] = None
    data_type: Optional[DataType] = None
    data: Optional[Any] = None
    user_info: dict
    response_language: Optional[str] = None


class ChatResponse(ResponseBase):
    pass


@dataclass
class UserInfo:
    auth_header: Optional[dict] = None
    data_type: Optional[str] = None
    date: str = datetime.now().astimezone().isoformat()
    restart: bool = False
    current_agent: Optional[str] = None
    interrupted_agent: Optional[str] = None
    user_input_language: Optional[str] = None


class BookingDetails(BaseModel):
    booking_slot_id: str
    vaccine: str
    clinic: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
    google_maps_url: Optional[str] = None


class RescheduleDetails(BaseModel):
    record_id: str  # old
    booking_slot_id: str  # new
    vaccine: str
    previous_clinic: Optional[str] = None
    previous_date: Optional[str] = None
    previous_time: Optional[str] = None
    new_clinic: Optional[str] = None
    new_date: Optional[str] = None
    new_time: Optional[str] = None
    google_maps_url: Optional[str] = None


class CancellationDetails(BaseModel):
    record_id: str
    vaccine: str
    clinic: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None
