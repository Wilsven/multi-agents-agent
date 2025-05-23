from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, EmailStr, field_validator

from app.schemas.address import AddressResponse
from app.schemas.base import HealthCondition, UserBase


class UserResponse(UserBase):
    address: AddressResponse | None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

    @field_validator("health_conditions", mode="before")
    def parse_health_conditions(cls, value):
        """
        Parse health_conditions from string to list of HealthCondition enums.
        """
        if value:
            if isinstance(value, str):
                # Handle empty string or "none" case
                if value.strip().lower() in {"", "none"}:
                    return None

                # Strip and split the string into a list
                health_conditions = [
                    condition.strip()
                    for condition in value.split(",")
                    if condition.strip() and condition.strip().lower() != "none"
                ]
                # Convert to enums (will raise if invalid)
                try:
                    return [
                        HealthCondition(condition) for condition in health_conditions
                    ]
                except ValueError as e:
                    raise ValueError(f"Invalid health condition in response: {e}")
            elif isinstance(value, list):
                return value
        # If NULL in database
        return None


class UserCreate(UserBase):
    postal_code: str
    password: str
    password_confirm: str


class UserUpdate(UserBase):
    postal_code: str


class UserCreateResponse(BaseModel):
    id: UUID
    email: EmailStr
    created_at: datetime

    class Config:
        from_attributes = True


class UserUpdateResponse(BaseModel):
    id: UUID
    email: EmailStr
    updated_at: datetime

    class Config:
        from_attributes = True
