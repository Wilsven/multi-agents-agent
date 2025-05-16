from enum import Enum


class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"


class QueryType(Enum):
    MANUAL = "manual"
    SUGGESTED = "suggested"
    FOLLOWUP = "followup"


class PersonaType(str, Enum):
    MYSELF = "myself"
    OTHERS = "others"
    GENERAL = "general"
    UNDEFINED = ""


class PersonaGender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    UNDEFINED = ""


class PersonaAgeType(str, Enum):
    YEARS = "years"
    MONTHS = "months"
