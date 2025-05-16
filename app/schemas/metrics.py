from pydantic import BaseModel


class MetricRequest(BaseModel):
    session_id: str
