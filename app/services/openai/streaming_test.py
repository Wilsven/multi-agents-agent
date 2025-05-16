import asyncio
import json
import time
from datetime import datetime

import httpx
import os
from app.schemas.chat import DataType, EventType
from dotenv import load_dotenv
# load_dotenv("../../../.azure/agents/.env")

timeout = httpx.Timeout(60.0, read=60.0)
BACKEND_URL = os.environ.get("BACKEND_MAIN_API_URL")

async def handle_stream(user_message: str, history: list, current_agent: str = None):
    async with httpx.AsyncClient(timeout=timeout) as client:
        start_time = time.time()

        # If you want it as a string
        start_time_string = datetime.fromtimestamp(start_time).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        async with client.stream(
            "POST",
            # "http://127.0.0.1:8001/voice/stream",
            "http://127.0.0.1:8001/chat/stream",
            # "http://127.0.0.1:8001/chat/stream_mcp",
            json={
                "message": user_message,
                "history": history,
                "agent_name": current_agent,
                "auth_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiNWIyMmM5MGEtMGEzNS00NGNhLThhNjEtYmIyYTgwY2U3ZGRkIiwicmVmcmVzaCI6ZmFsc2UsImV4cCI6MTc0NzM3NjY0OX0.IZbaCTKXU8WAhcy216WBROhBaBBEguhFimHd1B788XY",
                "session_id": test_session_id,
            },
        ) as response:
            async for line in response.aiter_lines():
                if not line.strip():
                    continue

                try:
                    chunk = json.loads(line)  # Convert JSON string to dict
                except json.JSONDecodeError:
                    print("❌ Invalid JSON:", line)
                    continue

                agent, history = handle_chat_response(chunk)

            end_time = time.time()
            end_time_string = datetime.fromtimestamp(end_time).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            print("=====starting time:", start_time_string, "======")
            print("=====ending time:", end_time_string, "======")
            print("Total time taken for calling:", end_time - start_time, "======")

            # test_get_tracing_info_by_time_range(start_time=start_time, end_time=end_time)
            await print_tracing_info_by_session_id(session_id=test_session_id)

            return agent, history


async def print_tracing_info_by_session_id(session_id):
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(
                "http://127.0.0.1:8001/metrics",
                json={
                    "session_id": session_id,
                },
            )
            response.raise_for_status()
            metrics_data = response.json()
            print("Metrics Data:", metrics_data)
        except httpx.HTTPStatusError as e:
            print(
                f"❌ HTTP error occurred: {e.response.status_code} - {e.response.text}"
            )
        except Exception as e:
            print(f"❌ An error occurred: {str(e)}")


def handle_chat_response(chunk: dict):
    # Handle event type first
    try:
        event_type = EventType(chunk["event_type"])
    except (KeyError, ValueError):
        event_type = None

    # Handle data type separately
    data_type = None
    if chunk.get("data_type"):
        try:
            data_type = DataType(chunk["data_type"])
        except ValueError:
            pass

    # Extract fields
    message = chunk.get("message")
    delta_message = chunk.get("delta_message")
    data = chunk.get("data")
    history = chunk.get("history")
    agent = chunk.get("agent_name")

    match event_type:
        case EventType.DELTA_TEXT_EVENT:
            print(f"{delta_message}", end="", flush=True)

        case EventType.COMPLETED_TEXT_EVENT:
            print(f"\n✅ [Complete Message] {message}")

        case EventType.NEW_AGENT_EVENT:
            print(f"\n🤖 Switched to agent: {agent}")

        case EventType.TOOL_CALL_EVENT:
            print(f"\n🛠️ Calling tool: {message}")

        case EventType.TOOL_CALL_OUTPUT_EVENT:
            if data_type:
                match data_type:
                    case DataType.BOOKING_DETAILS:
                        print(f"\n Booking Details:\n{json.dumps(data, indent=2)}")
                    case DataType.CANCEL_DETAILS:
                        print(f"\n Cancellation Details:\n{json.dumps(data, indent=2)}")
                    case DataType.RESCHEDULE_DETAILS:
                        print(f"\n Reschedule Details:\n{json.dumps(data, indent=2)}")
                    case _:
                        print(
                            f"\n🛠️ Tool Output ({data_type.value}):\n{json.dumps(data, indent=2)}"
                        )
            else:
                print(f"\n🛠️ Raw Tool Output:\n{json.dumps(data, indent=2)}")

        case EventType.TERMINATING_EVENT:
            print("\n🏁 Final History:")
            print(json.dumps(history, indent=2))

        case None:
            print(f"\n⚠️ Unrecognized Event. Chunk keys: {list(chunk.keys())}")

    return agent, history


if __name__ == "__main__":

    print("Starting streaming test...")
    # user_message = "recommend suitable vaccination for me"
    # user_message = "How to sleep better?"
    # user_message = "What is the nearest clinic for vaccination?"
    # user_message = "How to eat healthier?"
    # print("[User message (enter 'exit' to quit)] ", user_message)

    user_message = input("\n🙋‍♀️ [User message (enter 'exit' to quit)] ")
    history = None
    agent = None

    while True:
        # Call the function to handle streaming
        test_session_id = "test_ses_id" + str(int(time.time()))
        agent, history = asyncio.run(handle_stream(user_message, history, agent))

        user_message = input("\n🙋‍♀️ [User message (enter 'exit' to quit)] ")
        if user_message.lower() == "exit":
            break
