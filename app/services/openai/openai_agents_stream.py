import asyncio
import json
import os
import re
from dataclasses import asdict
from typing import AsyncGenerator, Optional

import httpx
from dotenv import load_dotenv
from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent

from agents import (
    AgentUpdatedStreamEvent,
    MessageOutputItem,
    RawResponsesStreamEvent,
    RunContextWrapper,
    RunItemStreamEvent,
    Runner,
    ToolCallItem,
    ToolCallOutputItem,
    TResponseInputItem,
)
from app.schemas.chat import ChatResponse, EventType, RequestType, UserInfo
from app.schemas.voice import VoiceResponse
from app.services.openai.agents import current_agent_mapping, triage_agent
from app.services.speech.text_to_speech import TextToSpeech


BACKEND_MAIN_API_URL = os.getenv("BACKEND_MAIN_API_URL")
# --------------------------
# Main function
# --------------------------
async def main(
    request_type: RequestType,
    user_msg: str,
    history: list | None,
    current_agent: str | None,
    auth_token: str,
    speech_client: Optional[TextToSpeech] = None,
) -> AsyncGenerator[ChatResponse | VoiceResponse, None]:

    detected_input_language = await get_user_input_language(user_msg)
    print("Detected language:", detected_input_language)

    # If you want to init wrapper with it
    wrapper = RunContextWrapper(
        context=UserInfo(
            auth_header={
                "Authorization": f"Bearer {auth_token}",
                "Content-Type": "application/json",
            },
            user_input_language=detected_input_language,
        )
    )

    # Init entry point agent
    if current_agent:
        agent = current_agent_mapping[current_agent]
    else:
        agent = triage_agent

    # If have exsting history, append new user message to it, else create new
    if history:
        history.append({"content": user_msg, "role": "user"})
    else:
        history: list[TResponseInputItem] = [{"content": user_msg, "role": "user"}]

    # Always init
    tool_output = None
    final_agents = {
        "general_questions_agent",
        "vaccination_records_agent",
        "recommender_agent",
        "manage_appointment_agent",
    }
    message = ""
    speech_chunk = ""

    result = Runner.run_streamed(agent, input=history, context=wrapper, max_turns=20)

    # Iterate through runner events
    async for event in result.stream_events():
        if isinstance(event, RawResponsesStreamEvent):
            """
            Raw response event: raw events directly from the LLM, in OpenAI Response API format
            For all the events, use `event.type` to retrieve the type of event
            """
            data = event.data
            if isinstance(
                data, ResponseTextDeltaEvent
            ):  # streaming text of a single LLM output
                message += data.delta  # collect the word by word output
                speech_chunk += data.delta
                response_dict = {
                    "event_type": EventType.DELTA_TEXT_EVENT,
                    "message": message,  # with latest delta message appended
                    "delta_message": data.delta,  # latest delta message
                    "data_type": None,
                    "data": None,
                    "history": None,
                    "agent_name": current_agent,
                    "user_info": asdict(wrapper.context),
                }

                # yield "Raw event TextDelta"
                if request_type == RequestType.CHAT_REQUEST:
                    yield ChatResponse(**response_dict)
                else:
                    # handle voice request
                    if speech_client and bool(
                        re.search(r"[.,!?。，！？:\n]\s", data.delta)
                    ):
                        audio_data = await speech_client.read_text(speech_chunk)
                        response_dict["audio_data"] = audio_data
                        speech_chunk = ""
                    yield VoiceResponse(**response_dict)

            elif isinstance(
                data, ResponseContentPartDoneEvent
            ):  # the end of a text output response
                message += "\n"
                response_dict = {
                    "event_type": EventType.COMPLETED_TEXT_EVENT,
                    "message": message,
                    "delta_message": None,
                    "data_type": None,
                    "data": None,
                    "history": None,
                    "agent_name": current_agent,
                    "user_info": asdict(wrapper.context),
                }

                # yield "Raw event ContentPartDone"
                if request_type == RequestType.CHAT_REQUEST:
                    yield ChatResponse(**response_dict)
                else:
                    # handle voice request
                    if speech_client:
                        audio_data = await speech_client.read_text(speech_chunk)
                        response_dict["audio_data"] = audio_data
                    speech_chunk = ""
                    yield VoiceResponse(**response_dict)

            else:  # other types of events
                pass

        elif isinstance(
            event, AgentUpdatedStreamEvent
        ):  # agent that is started / handed off to, e.g. triage_agent during init
            wrapper.context.current_agent = event.new_agent.name  # set in context
            current_agent = event.new_agent.name
            response_dict = {
                "event_type": EventType.NEW_AGENT_EVENT,
                "message": message,
                "data_type": None,
                "data": None,
                "history": None,
                "agent_name": current_agent,  # name of the agent that is handed off to
                "user_info": asdict(wrapper.context),
            }

            if request_type == RequestType.CHAT_REQUEST:
                yield ChatResponse(**response_dict)
            else:
                yield VoiceResponse(**response_dict)

        elif isinstance(
            event, RunItemStreamEvent
        ):  # Higher level event, inform me when an item has been fully generated, tool call
            """
            e.g. handoff: after all raw events, handoff_requested -> handoff_occured (include 'source_agent', and target agent 'raw_item.output.assistant')
            """
            if isinstance(event.item, ToolCallItem):
                response_dict = {
                    "event_type": EventType.TOOL_CALL_EVENT,
                    "message": event.item.raw_item.name,
                    "data_type": None,
                    "data": None,
                    "history": None,
                    "agent_name": event.item.agent.name,  # agent that called the tool
                    "user_info": asdict(wrapper.context),
                }

                if request_type == RequestType.CHAT_REQUEST:
                    yield ChatResponse(**response_dict)
                else:
                    yield VoiceResponse(**response_dict)

            # other type for evemt.item: ToolCallItem, ToolCallOutputItem, MessageOutputItem, HandoffCallItem, HandoffOutputItem
            elif isinstance(event.item, ToolCallOutputItem):  # tool call output
                if event.item.agent.name in {
                    "manage_appointment_agent",
                }:
                    tool_output = event.item.output

                    response_dict = {
                        "event_type": EventType.TOOL_CALL_OUTPUT_EVENT,
                        "message": None,
                        "data_type": wrapper.context.data_type,  # set by the individual agent during runtime
                        "data": tool_output,
                        "history": None,
                        "agent_name": event.item.agent.name,  # agent that called the tool
                        "user_info": asdict(wrapper.context),
                    }

                    # yield "Tool call output"
                    if request_type == RequestType.CHAT_REQUEST:
                        yield ChatResponse(**response_dict)
                    else:
                        yield VoiceResponse(**response_dict)

            elif isinstance(event.item, MessageOutputItem):
                pass

    current_agent = result.current_agent.name
    # If current agent is one of the final_agents, or restart flag set to True, change current agent to triage_agent
    if current_agent in final_agents or wrapper.context.restart:
        current_agent = "triage_agent"
        wrapper.context.restart = False

    # TODO: handle cases where halfmade booking cache should be removed
    history = result.to_input_list()

    last_message = history[-1]["content"][0]["text"]
    generated_response_language = await get_user_input_language(last_message)

    response_dict = {
        "event_type": EventType.TERMINATING_EVENT,  # the end of the conversation
        "message": None,
        "data_type": None,
        "data": None,
        "history": history,  # the consolidated history of the whole call
        "agent_name": current_agent,
        "user_info": asdict(wrapper.context),
        "response_language": generated_response_language,
    }

    # TODO: detect langauge

    if request_type == RequestType.CHAT_REQUEST:
        response = ChatResponse(**response_dict)
        yield response
    else:
        response = VoiceResponse(**response_dict)
        yield response


async def get_user_input_language(user_msg: str) -> str:
    url =  f"{BACKEND_MAIN_API_URL}/translate/get_language"
    headers = {"Content-Type": "application/json"}
    payload = {"text": user_msg}

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers=headers, timeout=10)

    data = response.json()
    if response.status_code != 200:
        raise Exception(f"Error: {response.status_code} - {response.text}")

    return data


async def simulate_stream():
    # convert user_msg to english
    user_data = {
        "username": "diana.lewis@example.com",
        "password": "guest",
    }  # 6 past records + 1 upcoming flu vaccine in late june

    login_data = {
        "grant_type": "password",  # Required
        "username": user_data["username"],  # Email used for login
        "password": user_data["password"],  # Password used for login
        "scope": "",  # Optional (empty value sent)
        "client_id": "",  # Optional (empty value sent)
        "client_secret": "",  # Optional (empty value sent)
    }

    async with httpx.AsyncClient(timeout=10.0) as httpclient:
        try:
            login = await httpclient.post(
                "http://127.0.0.1:8000/login", data=login_data
            )
        except Exception as e:
            print(f"Error making request: {e}")

    response = json.loads(login.text)
    auth_token = response["access_token"]

    # user_msg = "我要预约水痘疫苗."
    # user_msg = "I want to book a chickenpox vaccine."
    user_msg = "Tolong bantu saya menempah vaksin cacar air di Poliklinik Tampines pada bulan Mei"

    async for chunk in main(
        request_type=RequestType.CHAT_REQUEST,
        user_msg=user_msg,
        history=None,
        current_agent=None,
        auth_token=auth_token,
    ):
        print("================ Streamed Chunk ================")
        print(chunk)


if __name__ == "__main__":

    # asyncio.run(get_user_input_language("9p834urhnf0i43u32"))
    asyncio.run(simulate_stream())
