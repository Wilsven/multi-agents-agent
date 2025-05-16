import base64
import json
import os
import urllib.parse
import uuid
from datetime import datetime, timedelta
from typing import Dict, List

import httpx
import pytz
import requests
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from openai.types.chat import ChatCompletion
from openai_messages_token_helper import build_messages

from agents import RunContextWrapper, function_tool
from app.schemas.chat import (
    BookingDetails,
    CancellationDetails,
    RescheduleDetails,
    UserInfo,
)
from app.schemas.healthhubai import (
    MessageRole,
    PersonaAgeType,
    PersonaGender,
    PersonaType,
    QueryType,
)

# --------------------------
# Load environment variables
# --------------------------
# load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../../../../.env"))
OPENAI_CHATGPT_MODEL = os.environ["AZURE_OPENAI_CHATGPT_MODEL"]
AZURE_OPENAI_SERVICE = os.environ["AZURE_OPENAI_SERVICE"]
AZURE_OPENAI_CHATGPT_DEPLOYMENT = os.environ["AZURE_OPENAI_CHATGPT_DEPLOYMENT"]
AZURE_HHAI_CHAT_ENDPOINT = os.environ["AZURE_HHAI_CHAT_ENDPOINT"]
AZURE_HHAI_CHAT_SESSION_ID = os.environ["AZURE_HHAI_CHAT_SESSION_ID"]
BACKEND_MAIN_API_URL = os.environ["BACKEND_MAIN_API_URL"]

SEED = 1234
RESPONSE_TOKEN_LIMIT = 512
CHATGPT_TOKEN_LIMIT = 128000

credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(
    credential, "https://cognitiveservices.azure.com/.default"
)


client = AsyncAzureOpenAI(
    api_version="2024-10-21",
    azure_endpoint=f"https://{AZURE_OPENAI_SERVICE}.openai.azure.com",
    azure_ad_token_provider=token_provider,
    azure_deployment=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
)


# --------------------------
# Tools Definition
# --------------------------
async def get_past_records(
    wrapper: RunContextWrapper[UserInfo], requested_vaccine: str | None = None
) -> list | str:
    """
    Helper function for get_vaccination_history_tool and get_latest_vaccination_tool

    Args:
        requested_vaccine: Optional parameter to filter past records by vaccine type. If none, returns most recent records for all vaccine types.
    """
    # Get the booked slots from user
    async with httpx.AsyncClient(timeout=10.0) as httpclient:
        try:
            vaccination_history_records = await httpclient.get(
                f"{BACKEND_MAIN_API_URL}/records",
                headers=wrapper.context.context.auth_header,
            )
            if vaccination_history_records.status_code == 404:
                return "No records found."
        except Exception as e:
            print(f"Error making request: {e}")

    vaccination_history_records = json.loads(vaccination_history_records.text)

    most_recent_records = {}

    for record in vaccination_history_records:
        booking_slot_id = record["booking_slot_id"]
        async with httpx.AsyncClient(timeout=10.0) as httpclient:
            try:
                booking_slot = await httpclient.get(
                    f"{BACKEND_MAIN_API_URL}/bookings/{booking_slot_id}",
                    headers=wrapper.context.context.auth_header,
                )
                if booking_slot.status_code == 404:
                    return "Missing booking slot."
            except Exception as e:
                print(f"Error making request: {e}")

        booking_slot = json.loads(booking_slot.text)
        del record["created_at"]
        record["vaccine_name"] = booking_slot["vaccine"]["name"]
        record["slot_date"] = booking_slot["datetime"]
        record["polyclinic"] = booking_slot["polyclinic"]["name"]
        record_date = datetime.fromisoformat(record["slot_date"]).replace(
            tzinfo=pytz.UTC
        )
        current_date = datetime.fromisoformat(
            f"{wrapper.context.context.date}"
        ).replace(tzinfo=pytz.UTC)

        # Filter for past records
        if record_date < current_date:
            vaccine_type = record["vaccine_name"]
            # Option to filter by vaccine type, if requested_vaccine is provided
            if requested_vaccine and vaccine_type != requested_vaccine:
                continue

            # Update most_recent_records with latest vaccination record for each vaccine type
            if (
                vaccine_type not in most_recent_records
                or record_date
                > datetime.fromisoformat(
                    most_recent_records[vaccine_type]["slot_date"]
                ).replace(tzinfo=pytz.UTC)
            ):
                most_recent_records[vaccine_type] = record

    augmented_records = list(most_recent_records.values())

    return augmented_records


@function_tool
async def get_vaccination_history_tool(
    wrapper: RunContextWrapper[UserInfo],
) -> list | str:
    """
    Gets list of past vaccinations of the user and the respective recommended frequencies of vaccines.
    """
    result = await get_past_records(wrapper)

    return result


@function_tool
async def get_latest_vaccination_tool(
    wrapper: RunContextWrapper[UserInfo], requested_vaccine: str
) -> list | str:
    """
    Gets most recent vaccinations of the user for the requested vaccine type and its recommended vaccination frequency.

    Args:
      requested_vaccine: User input of vaccine type found from chat history.
    """
    result = await get_past_records(wrapper, requested_vaccine)

    return result


@function_tool
async def get_upcoming_appointments_tool(
    wrapper: RunContextWrapper[UserInfo],
) -> list | str:
    """
    Gets list of current bookings from the user, which can be used to check if user already has an existing booking for the vaccine requested.
    """
    # Get the booked slots from user
    async with httpx.AsyncClient(timeout=10.0) as httpclient:
        try:
            vaccination_history_records = await httpclient.get(
                f"{BACKEND_MAIN_API_URL}/records",
                headers=wrapper.context.context.auth_header,
            )
            if vaccination_history_records.status_code == 404:
                return "No records found."
        except Exception as e:
            print(f"Error making request: {e}")
    vaccination_history_records = json.loads(vaccination_history_records.text)

    # Update the slots with respective vaccine names and date taken
    augmented_records = []
    for record in vaccination_history_records:
        booking_slot_id = record["booking_slot_id"]
        async with httpx.AsyncClient(timeout=10.0) as httpclient:
            try:
                booking_slot = await httpclient.get(
                    f"{BACKEND_MAIN_API_URL}/bookings/{booking_slot_id}",
                    headers=wrapper.context.context.auth_header,
                )
                if booking_slot.status_code == 404:
                    return "Missing booking slot."
            except Exception as e:
                print(f"Error making request: {e}")
        booking_slot = json.loads(booking_slot.text)
        del record["created_at"]
        record["vaccine_name"] = booking_slot["vaccine"]["name"]
        record["slot_date"] = booking_slot["datetime"]
        record["polyclinic"] = booking_slot["polyclinic"]["name"]
        if datetime.fromisoformat(record["slot_date"]).replace(
            tzinfo=pytz.UTC
        ) >= datetime.fromisoformat(f"{wrapper.context.context.date}").replace(
            tzinfo=pytz.UTC
        ):
            augmented_records.append(record)

    return augmented_records


@function_tool
async def recommend_vaccines_tool(wrapper: RunContextWrapper[UserInfo]) -> str:
    """
    Get vaccine recommendations for user based on their demographic.
    """
    async with httpx.AsyncClient(timeout=10.0) as httpclient:
        try:
            recommendations = await httpclient.get(
                f"{BACKEND_MAIN_API_URL}/vaccines/recommendations",
                headers=wrapper.context.context.auth_header,
            )
            if recommendations.status_code == 404:
                return "Unable to get recommendations for user."
        except Exception as e:
            print(f"Error making request: {e}")
    recommendations = json.loads(recommendations.text)
    return json.dumps(recommendations)


@function_tool
async def standardise_vaccine_name_tool(
    wrapper: RunContextWrapper[UserInfo], requested_vaccine: str
) -> dict | str:
    """
    Always use this tool when the step requires it.

    Args:
        requested_vaccine: User input of vaccine type found from chat history.
    """
    standard_name_prompt = f"""
    The input may use informal name, and your task is to map it to the correct official name. For example, the input "flu vaccine" should be mapped to "Influenza (INF)".

    Find the closest match of {requested_vaccine} to the list below:
    - Influenza (INF)
    - Pneumococcal Conjugate (PCV13)
    - Human Papillomavirus (HPV)
    - Tetanus, Diphtheria, Pertussis (Tdap)
    - Hepatitis B (HepB)
    - Measles, Mumps, Rubella (MMR)
    - Varicella (VAR)

    If there is a match, return the value in the list exactly.
    Else, return "Handoff to recommender_agent"

    Official vaccine name:
    """

    messages = build_messages(
        model=OPENAI_CHATGPT_MODEL,
        system_prompt=standard_name_prompt,
        max_tokens=CHATGPT_TOKEN_LIMIT - RESPONSE_TOKEN_LIMIT,
    )

    chat_completion: ChatCompletion = await client.chat.completions.create(
        model=AZURE_OPENAI_CHATGPT_DEPLOYMENT,
        messages=messages,
        temperature=0,
        max_tokens=RESPONSE_TOKEN_LIMIT,
        n=1,
        stream=False,
        seed=SEED,
    )

    llm_output = chat_completion.choices[0].message.content
    if llm_output != "None":
        response_dict = {
            "vaccine_name": llm_output,
        }
        return response_dict
    else:
        return llm_output


async def get_location_info(location_name: str) -> dict | None:
    """
    OneMap Search API to get the postal code, address, latitude, and longitude of a location.
    """
    url = f"https://www.onemap.gov.sg/api/common/elastic/search?searchVal={location_name}&returnGeom=Y&getAddrDetails=Y&pageNum=1"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if data["found"] > 0:
            result = data["results"][0]
            return {
                "postal_code": result["POSTAL"],
                "address": result["ADDRESS"],
                "latitude": result["LATITUDE"],
                "longitude": result["LONGITUDE"],
            }
    return None


@function_tool
async def get_clinics_near_location_tool(
    wrapper: RunContextWrapper[UserInfo], location_name: str
) -> dict | str:
    """
    Always use this tool when the step requires it.
    Returns polyclinics closest to the location provided by the user.

    Args:
        location_name: Takes the location specified by the user
    """
    # Retrieve address information
    location_result = await get_location_info(location_name)
    latitude = float(location_result["latitude"])
    longitude = float(location_result["longitude"])

    try:
        async with httpx.AsyncClient(timeout=10.0) as httpclient:
            get_recommended_polyclinic = await httpclient.get(
                url=f"{BACKEND_MAIN_API_URL}/clinics/nearest-by-location",
                headers=wrapper.context.context.auth_header,
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "clinic_type": "polyclinic",
                    "clinic_limit": 3,
                },
            )
    except Exception as e:
        print(f"Error making request: {e}")

    recommended_polyclinics = json.loads(get_recommended_polyclinic.text)
    return json.dumps(recommended_polyclinics)


@function_tool
async def get_clinic_name_response_helper_tool(
    wrapper: RunContextWrapper[UserInfo], clinic_name: str
) -> dict | str:
    """
    Use this tool when polyclinic name is foud in chat history.

    Args:
        clinic_name: Takes the clinic name found in chat history (e.g. "Toa Payoh Polyclinic")
    """
    response_dict = {
        "clinic": clinic_name,
    }

    return response_dict


@function_tool
async def get_clinics_near_home_tool(
    wrapper: RunContextWrapper[UserInfo],
) -> dict | str:
    """
    Use this tool when polyclinic name is not found in chat history and user did not specify a location.
    Returns polyclinics closest to the user's home address.
    """
    # Recommend polyclinics near home
    try:
        async with httpx.AsyncClient(timeout=10.0) as httpclient:
            get_recommended_polyclinic = await httpclient.get(
                url=f"{BACKEND_MAIN_API_URL}/clinics/nearest-by-home",
                headers=wrapper.context.context.auth_header,
                params={
                    "clinic_type": "polyclinic",
                    "clinic_limit": 3,
                },
            )
    except Exception as e:
        print(f"Error making request: {e}")

    recommended_polyclinics = json.loads(get_recommended_polyclinic.text)
    return json.dumps(recommended_polyclinics)


@function_tool
async def get_available_slots_tool(
    wrapper: RunContextWrapper[UserInfo],
    vaccine_name: str,
    clinic: str,
    start_date: str = None,
    end_date: str = None,
) -> List[Dict] | str:
    """
    Get available slots for a vaccine at a specific clinic, over date ranges if specified.

    Args:
        vaccine_name: Official name of vaccine type
        clinic: The name of clinic
        start_date: Optional parameter. Include only if start date is provided by user. (Date in ISO format)
        end_date: Optional parameter. Include only if end date is provided by user. (Date in ISO format)
    """
    if not start_date:
        start_date = wrapper.context.context.date
    if not end_date:
        end_date = (datetime.fromisoformat(start_date) + timedelta(days=3)).isoformat()

    try:
        async with httpx.AsyncClient(timeout=10.0) as httpclient:
            get_slots = await httpclient.get(
                url=f"{BACKEND_MAIN_API_URL}/bookings/available",
                headers=wrapper.context.context.auth_header,
                params={
                    "vaccine_name": vaccine_name,
                    "polyclinic_name": clinic,
                    "start_datetime": start_date,
                    "end_datetime": end_date,
                    "timeslot_limit": 3,
                },
            )
            if get_slots.status_code == 404:
                return "No avaialable slots for current date range or clinic."
    except Exception as e:
        print(f"Error making request: {e}")
    result = json.loads(get_slots.text)
    return json.dumps(result)


@function_tool
async def recommend_gps_tool(wrapper: RunContextWrapper[UserInfo]) -> str:
    """
    Get nearest GPs to user's home address, if there are no available slots found at selected polyclinic.
    """
    # Get nearest 3 GPs to user's postal code
    try:
        async with httpx.AsyncClient(timeout=10.0) as httpclient:
            get_recommended_gp = await httpclient.get(
                url=f"{BACKEND_MAIN_API_URL}/clinics/nearest-by-home",
                headers=wrapper.context.context.auth_header,
                params={
                    "clinic_type": "gp",
                    "clinic_limit": 3,
                },
            )
    except Exception as e:
        print(f"Error making request: {e}")

    recommended_gps = json.loads(get_recommended_gp.text)
    return json.dumps(recommended_gps)


async def construct_google_maps_url(
    wrapper: RunContextWrapper[UserInfo],
    destination_name: str,
    travel_mode: str = "transit",
) -> str:
    """
    Helper function for new_appointment_tool and reschedule_appointment_tool

    Args:
        destination_name: The destination polyclinic to be used in the Google Maps URL
        travel_mode: The mode of travel to be used in the Google Maps URL
    """
    # Get user's postal code
    async with httpx.AsyncClient(timeout=10.0) as httpclient:
        try:
            user_profile = await httpclient.get(
                "{BACKEND_MAIN_API_URL}/users",
                headers=wrapper.context.context.auth_header,
            )
        except Exception as e:
            print(f"Error making request: {e}")

    user_profile = json.loads(user_profile.text)
    origin_postal_code = user_profile["address"]["postal_code"]

    # Construct Google Maps URL
    base_url = "https://www.google.com/maps/dir/?api=1"
    params = {
        "origin": origin_postal_code,
        "destination": destination_name,
        "travelmode": travel_mode,
    }
    query_string = urllib.parse.urlencode(params)
    return f"{base_url}&{query_string}"


@function_tool
async def new_appointment_tool(
    wrapper: RunContextWrapper[UserInfo], slot_id: str
) -> BookingDetails:
    """
    Gets booking details of a new appointment.

    Args:
        slot_id: The 'id' field for slot to be booked
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as httpclient:
            slot = await httpclient.get(
                url=f"{BACKEND_MAIN_API_URL}/bookings/{slot_id}",
                headers=wrapper.context.context.auth_header,
            )
    except Exception as e:
        return f"Error making request: {e}"

    slot = json.loads(slot.text)
    dt_object = datetime.fromisoformat(slot["datetime"].replace("Z", "+00:00"))

    wrapper.context.context.data_type = "booking_details"
    response_dict = {
        "booking_slot_id": slot_id,
        "vaccine": slot["vaccine"]["name"],
        "clinic": slot["polyclinic"]["name"],
        "date": str(dt_object.date()),
        "time": str(dt_object.time()),
        "google_maps_url": await construct_google_maps_url(
            wrapper, slot["polyclinic"]["name"]
        ),
    }
    response = BookingDetails(**response_dict).model_dump()

    return json.dumps(response)


@function_tool
async def change_appointment_tool(
    wrapper: RunContextWrapper[UserInfo],
    record_id: str,
    new_slot_id: str,
) -> BookingDetails:
    """
    Gets booking details of the existing appointment and the new appointment it is rescheduled to.

    Args:
        record_id (str): The id for the vaccination appointment record to remove
        new_slot_id (str): The id of slot to reschedule a current slot to
    """
    # ---------------------------
    # Get old appointment details
    # ---------------------------
    try:
        async with httpx.AsyncClient(timeout=10.0) as httpclient:
            old_vaccination_record = await httpclient.get(
                f"{BACKEND_MAIN_API_URL}/records/{record_id}",
                headers=wrapper.context.context.auth_header,
            )
    except Exception as e:
        return f"Error making request: {e}"
    old_vaccination_record = json.loads(old_vaccination_record.text)

    # Update the cancelled slot with respective vaccine names and date taken
    old_booking_slot_id = old_vaccination_record["booking_slot_id"]
    try:
        async with httpx.AsyncClient(timeout=10.0) as httpclient:
            old_booking_slot = await httpclient.get(
                f"{BACKEND_MAIN_API_URL}/bookings/{old_booking_slot_id}",
                headers=wrapper.context.context.auth_header,
            )
    except Exception as e:
        return f"Error making request: {e}"

    old_booking_slot = json.loads(old_booking_slot.text)
    old_dt_object = datetime.fromisoformat(
        old_booking_slot["datetime"].replace("Z", "+00:00")
    )

    # ---------------------------
    # Get new appointment details
    # ---------------------------
    try:
        async with httpx.AsyncClient(timeout=10.0) as httpclient:
            new_slot = await httpclient.get(
                url=f"{BACKEND_MAIN_API_URL}/bookings/{new_slot_id}",
                headers=wrapper.context.context.auth_header,
            )
    except Exception as e:
        return f"Error making request: {e}"

    new_slot = json.loads(new_slot.text)
    new_dt_object = datetime.fromisoformat(new_slot["datetime"].replace("Z", "+00:00"))

    wrapper.context.context.data_type = "reschedule_details"

    response_dict = {
        "record_id": record_id,
        "booking_slot_id": new_slot_id,
        "vaccine": new_slot["vaccine"]["name"],
        "previous_clinic": old_booking_slot["polyclinic"]["name"],
        "previous_date": str(old_dt_object.date()),
        "previous_time": str(old_dt_object.time()),
        "new_clinic": new_slot["polyclinic"]["name"],
        "new_date": str(new_dt_object.date()),
        "new_time": str(new_dt_object.time()),
        "google_maps_url": await construct_google_maps_url(
            wrapper, new_slot["polyclinic"]["name"]
        ),
    }
    response = RescheduleDetails(**response_dict).model_dump()

    return json.dumps(response)


@function_tool
async def cancel_appointment_tool(wrapper: RunContextWrapper[UserInfo], record_id: str):
    """
    Gets booking details of the existing appointment to be cancelled.

    Args:
        record_id (str): The id for the vaccination appointment record to remove
    """
    # Get the record_id from user, to get confirmation to cancel the appointment
    try:
        async with httpx.AsyncClient(timeout=10.0) as httpclient:
            vaccination_record = await httpclient.get(
                f"{BACKEND_MAIN_API_URL}/records/{record_id}",
                headers=wrapper.context.context.auth_header,
            )
    except Exception as e:
        return f"Error making request: {e}"
    vaccination_record = json.loads(vaccination_record.text)

    # Update the cancelled slot with respective vaccine names and date taken
    booking_slot_id = vaccination_record["booking_slot_id"]
    try:
        async with httpx.AsyncClient(timeout=10.0) as httpclient:
            booking_slot = await httpclient.get(
                f"{BACKEND_MAIN_API_URL}/bookings/{booking_slot_id}",
                headers=wrapper.context.context.auth_header,
            )
    except Exception as e:
        return f"Error making request: {e}"

    booking_slot = json.loads(booking_slot.text)
    dt_object = datetime.fromisoformat(booking_slot["datetime"].replace("Z", "+00:00"))

    wrapper.context.context.data_type = "cancel_details"

    response_dict = {
        "record_id": record_id,
        "vaccine": booking_slot["vaccine"]["name"],
        "clinic": booking_slot["polyclinic"]["name"],
        "date": str(dt_object.date()),
        "time": str(dt_object.time()),
    }
    response = CancellationDetails(**response_dict).model_dump()

    return json.dumps(response)


@function_tool
async def healthhub_ai_tool(
    wrapper: RunContextWrapper[UserInfo], user_query: str
) -> str:
    """
    Forward health-related queries to HealthHub AI chatbot.
    This tool allows the agent to obtain information for health related queries.

    Args:
        user_query: The health-related question to send to the chatbot
    """
    print(f"[TOOL CALL] healthhub_ai_tool called with user_query: {user_query}")
    try:
        # Base64 encoding function
        def to_base64(text):
            return base64.b64encode(text.encode("utf-8")).decode("utf-8")

        tool_persona = {
            "id": "tool_persona",
            "name": "HHAI Tool",
            "persona_type": PersonaType.GENERAL.value,
            "gender": PersonaGender.MALE.value,
            "age": 30,
            "age_type": PersonaAgeType.YEARS.value,
            "existing_conditions": "none",
            "persona_config": {},
        }

        # Headers
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/plain, */*",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "X-Session-Id": AZURE_HHAI_CHAT_SESSION_ID,
        }

        data = {
            "persona": tool_persona,
            "query": {
                "role": to_base64(MessageRole.USER.value),
                "content": to_base64(user_query),
                "query_type": to_base64(QueryType.MANUAL.value),
                "message_id": to_base64(str(uuid.uuid4())),
            },
            "page_language": "english",  # TODO HANDLE MORE LANGUAGES
        }

        response_text = ""
        async with httpx.AsyncClient(timeout=10.0) as client:
            async with client.stream(
                "POST",
                AZURE_HHAI_CHAT_ENDPOINT,
                headers=headers,
                json=data,
            ) as response:
                async for chunk in response.aiter_text():
                    for line in chunk.splitlines():
                        if not line.strip():
                            continue

                        buffer = line.strip()
                        while buffer:
                            try:
                                # Parse ONE JSON object from the buffer
                                obj, idx = json.JSONDecoder().raw_decode(buffer)
                                buffer = buffer[idx:].lstrip()  # Remove parsed data

                                # Extract message
                                msg = obj.get("response_message", "")
                                if msg:
                                    response_text += msg

                            except json.JSONDecodeError as e:
                                print(f"[Partial JSON]: {buffer[:50]}... (error: {e})")
                                break  # Incomplete JSON; wait for next chunk

        return response_text

    except Exception as e:
        return f"HealthHub chatbot unavailable: {str(e)}"
