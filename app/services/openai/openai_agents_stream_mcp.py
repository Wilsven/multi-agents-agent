import asyncio
import os
import re
from dataclasses import asdict
from typing import AsyncGenerator, Optional

from dotenv import load_dotenv
from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent

from agents import (
    Agent,
    AgentUpdatedStreamEvent,
    MessageOutputItem,
    RawResponsesStreamEvent,
    RunContextWrapper,
    RunItemStreamEvent,
    Runner,
    ToolCallItem,
    ToolCallOutputItem,
    TResponseInputItem,
    handoff,
    set_default_openai_key,
    set_tracing_disabled,
)
from agents.mcp import MCPServerSse
from app.schemas.chat import ChatResponse, EventType, RequestType, UserInfo
from app.schemas.voice import VoiceResponse
from app.services.openai.agents import (
    appointments_agent,
    check_available_slots_agent,
    double_booking_check_agent,
    handle_vaccine_names_agent,
    identify_clinic_agent,
    interrupt_handler_agent_prompt,
    manage_appointment_agent,
    modify_existing_appointment_agent,
    recommended_vaccine_check_agent,
    recommender_agent,
    vaccination_history_check_agent,
    vaccination_records_agent,
)
from app.services.speech.text_to_speech import TextToSpeech

# --------------------------
# Load environment variables
# --------------------------
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../../../../.env"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_MCP_HHAI_ENDPOINT = os.getenv(
    "AZURE_MCP_HHAI_ENDPOINT",
)

AZURE_MCP_HHAI_API_KEY = os.getenv(
    "AZURE_MCP_HHAI_API_KEY",
)

set_tracing_disabled(True)
set_default_openai_key(f"{OPENAI_API_KEY}")


# --------------------------
# Main MCP function
# --------------------------
async def main_mcp(
    request_type: RequestType,
    user_msg: str,
    history: list | None,
    current_agent: str | None,
    auth_token: str,
    speech_client: Optional[TextToSpeech] = None,
) -> AsyncGenerator[ChatResponse | VoiceResponse, None]:
    hhai_mcp_server = MCPServerSse(
        params={
            "url": f"{AZURE_MCP_HHAI_ENDPOINT}/sse",
            "headers": {"x-api-key": AZURE_MCP_HHAI_API_KEY},
        },
        cache_tools_list=True,
    )

    async with hhai_mcp_server:

        general_questions_agent_mcp = Agent(
            name="general_questions_agent_mcp",
            instructions=(
                "You are a proxy for HealthHub AI and must strictly follow these rules: for any health, wellness, or lifestyle query, always call the healthhub_ai_tool and return its response verbatim,"
                "For any health, wellness, or lifestyle questions, always use the healthhub_ai_tool to obtain the answer. "
                "Do not answer health or wellness questions using your own knowledge—always call the healthhub_ai_tool for these. "
                "If the user's question is NOT related to healthcare, health, wellness, or lifestyle, respond with: "
                "'I'm sorry, I am only able to answer healthcare or health-related queries.' "
                "When a user sends a short message like 'yes', 'okay', or similar, always check the previous chat history to determine if it is a follow-up to a health-related query. "
                "If it is, continue the conversation appropriately using the healthhub_ai_tool. "
            ),
            model="gpt-4o-mini",
            mcp_servers=[hhai_mcp_server],
        )

        triage_agent_mcp = Agent(
            name="triage_agent_mcp",
            instructions=(
                "You are a helpful assistant that directs user queries to the appropriate agents. Do not ask or answer any questions yourself, only handoff to other agents."
                "Take the conversation history as context when deciding who to handoff to next."
                "If they want to book vaccination appointments, but did not mention which vaccine they want, handoff to the recommender_agent."
                "If they mentioned their desired vaccine and would like to book an appointment, handoff to appointments_agent."
                "If they ask for vaccination reccomendations, handoff to recommender_agent."
                "If they ask about vaccination records, like asking about their past vaccinations, handoff to vaccination_records_agent."
                "Otherwise, handoff to general_questions_agent_mcp."
            ),
            handoffs=[
                appointments_agent,
                recommender_agent,
                vaccination_records_agent,
                general_questions_agent_mcp,
            ],
            model="gpt-4o-mini",
        )

        interrupt_handler_agent_mcp = Agent(
            instructions=interrupt_handler_agent_prompt,
            name="interrupt_handler_agent_mcp",
            model="gpt-4o-mini",
            tools=[
                recommender_agent.as_tool(
                    tool_name="recommend_vaccines_tool",
                    tool_description="Gets recommended vaccines for user.",
                ),
                vaccination_records_agent.as_tool(
                    tool_name="vaccine_records_tool",
                    tool_description="Gets user's vaccination records.",
                ),
                general_questions_agent_mcp.as_tool(
                    tool_name="general_questions_tool",
                    tool_description="Answers all general questions.",
                ),
            ],
            handoff_description="Handoff to this agent when the user is not responding to your question.",
        )

        # --------------------------
        # Add backlines
        # --------------------------
        check_available_slots_agent.handoffs.append(manage_appointment_agent)
        double_booking_check_agent.handoffs.append(manage_appointment_agent)
        double_booking_check_agent.handoffs.append(triage_agent_mcp)
        vaccination_history_check_agent.handoffs.append(triage_agent_mcp)
        check_available_slots_agent.handoffs.append(triage_agent_mcp)

        # --------------------------
        # Current agent mapping
        # --------------------------
        current_agent_mapping = {
            "triage_agent_mcp": triage_agent_mcp,
            "interrupt_handler_agent_mcp": interrupt_handler_agent_mcp,
            "double_booking_check_agent": double_booking_check_agent,
            "recommended_vaccine_check_agent": recommended_vaccine_check_agent,
            "vaccination_history_check_agent": vaccination_history_check_agent,
            "identify_clinic_agent": identify_clinic_agent,
            "check_available_slots_agent": check_available_slots_agent,
            "modify_existing_appointment_agent": modify_existing_appointment_agent,
        }

        # -----------------------------------------
        # Handle handoff to interrupt_handler_agent_mcp
        # -----------------------------------------
        async def on_interrupt_handoff(wrapper: RunContextWrapper[UserInfo]):
            # Set the current agent that is being interrupted as the interrupted agent
            interrupted_agent_name = wrapper.context.context.current_agent
            wrapper.context.context.interrupted_agent = interrupted_agent_name
            # Reset the handoffs for interrupt_handler_agent_mcp to handle for specific interruption
            interrupt_handler_agent_mcp.handoffs = [
                triage_agent_mcp,
                current_agent_mapping.get(interrupted_agent_name),
            ]

        # Add handoff to interrupt_handler_agent_mcp
        double_booking_check_agent.handoffs.append(
            handoff(agent=interrupt_handler_agent_mcp, on_handoff=on_interrupt_handoff)
        )
        check_available_slots_agent.handoffs.append(
            handoff(agent=interrupt_handler_agent_mcp, on_handoff=on_interrupt_handoff)
        )
        identify_clinic_agent.handoffs.append(
            handoff(agent=interrupt_handler_agent_mcp, on_handoff=on_interrupt_handoff)
        )
        vaccination_history_check_agent.handoffs.append(
            handoff(agent=interrupt_handler_agent_mcp, on_handoff=on_interrupt_handoff)
        )
        recommended_vaccine_check_agent.handoffs.append(
            handoff(agent=interrupt_handler_agent_mcp, on_handoff=on_interrupt_handoff)
        )
        modify_existing_appointment_agent.handoffs.append(
            handoff(agent=interrupt_handler_agent_mcp, on_handoff=on_interrupt_handoff)
        )

        # Init RunContextWrapper with auth token
        wrapper = RunContextWrapper(
            context=UserInfo(
                auth_header={
                    "Authorization": f"Bearer {auth_token}",
                    "Content-Type": "application/json",
                }
            )
        )

        # Init entry point agent
        if current_agent:
            current_agent_mapping = {
                "triage_agent_mcp": triage_agent_mcp,
                "interrupt_handler_agent_mcp": interrupt_handler_agent_mcp,
                "handle_vaccine_names_agent": handle_vaccine_names_agent,
                "double_booking_check_agent": double_booking_check_agent,
                "recommended_vaccine_check_agent": recommended_vaccine_check_agent,
                "vaccination_history_check_agent": vaccination_history_check_agent,
                "identify_clinic_agent": identify_clinic_agent,
                "check_available_slots_agent": check_available_slots_agent,
                "modify_existing_appointment_agent": modify_existing_appointment_agent,
            }
            agent = current_agent_mapping[current_agent]
        else:
            agent = triage_agent_mcp

        # If have exsting history, append new user message to it, else create new
        if history:
            history.append({"content": user_msg, "role": "user"})
        else:
            history: list[TResponseInputItem] = [{"content": user_msg, "role": "user"}]

        # Always init
        tool_output = None
        final_agents = {
            "general_questions_agent_mcp",
            "vaccination_records_agent",
            "recommender_agent",
            "manage_appointment_agent",
        }
        message = ""
        speech_chunk = ""

        result = Runner.run_streamed(
            agent, input=history, context=wrapper, max_turns=20
        )

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
                        if bool(re.search(r"[.,!?。，！？]\s", data.delta)):
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
            current_agent = "triage_agent_mcp"
            wrapper.context.restart = False

        # TODO: handle cases where halfmade booking cache should be removed
        history = result.to_input_list()
        response_dict = {
            "event_type": EventType.TERMINATING_EVENT,  # the end of the conversation
            "message": None,
            "data_type": None,
            "data": None,
            "history": history,  # the consolidated history of the whole call
            "agent_name": current_agent,
            "user_info": asdict(wrapper.context),
        }

        if request_type == RequestType.CHAT_REQUEST:
            response = ChatResponse(**response_dict)
            yield response
        else:
            pass
            # response = VoiceResponse(**response_dict)
            # yield response


async def simulate_stream_general():
    async for chunk in main_mcp(
        request_type=RequestType.CHAT_REQUEST,
        # user_msg="How to sleep well.",
        user_msg="Check my vaccination history.",
        history=None,
        current_agent=None,
        user_info=None,
    ):
        print("================ Streamed Chunk ================")
        print(chunk)


if __name__ == "__main__":

    asyncio.run(simulate_stream_general())
