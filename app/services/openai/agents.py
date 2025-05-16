import os

from dotenv import load_dotenv

from agents import (
    Agent,
    ModelSettings,
    RunContextWrapper,
    handoff,
    set_default_openai_key,
    set_tracing_disabled,
)
from app.schemas.chat import UserInfo
from app.services.openai.tools import (
    cancel_appointment_tool,
    change_appointment_tool,
    get_available_slots_tool,
    get_clinic_name_response_helper_tool,
    get_clinics_near_home_tool,
    get_clinics_near_location_tool,
    get_latest_vaccination_tool,
    get_upcoming_appointments_tool,
    get_vaccination_history_tool,
    healthhub_ai_tool,
    new_appointment_tool,
    recommend_gps_tool,
    recommend_vaccines_tool,
    standardise_vaccine_name_tool,
)

# --------------------------
# Load environment variables
# --------------------------
# load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../../../../.env"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

set_tracing_disabled(True)
set_default_openai_key(f"{OPENAI_API_KEY}")


# --------------------------
# Agents Definition
# --------------------------
general_questions_agent = Agent(
    name="general_questions_agent",
    instructions=(
        "You are a proxy for HealthHub AI and must strictly follow these rules: always call the healthhub_ai_tool and return its response verbatim,"
        "Do not answer using your own knowledge. Always use the healthhub_ai_tool."
    ),
    tools=[healthhub_ai_tool],
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="healthhub_ai_tool"),
)


vaccination_records_agent = Agent[UserInfo](
    name="vaccination_records_agent",
    instructions=(
        """
    You are tasked with retrieving the records of the user.
    Use get_vaccination_history_tool to retrieve the records.
    Output the results.
    """
    ),
    tools=[get_vaccination_history_tool],
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="get_vaccination_history_tool"),
)


def recommender_agent_prompt(
    context_wrapper: RunContextWrapper[UserInfo], agent: Agent[UserInfo]
) -> str:
    context: UserInfo = context_wrapper.context.context
    return f"""
    You are tasked with giving vaccine recommendations for the user based on their vaccination history.
    Today's date: {context.date}
    Follow the steps in order:
    1. Use the recommend_vaccines_tool to get the vaccine recommendations based on their demographic.
    2. Use the get_vaccination_history_tool to retrieve the past vaccination records of the user.
    3. Use the get_upcoming_appointments_tool to retrieve the upcoming appointments user has.
    4. Use the information from the tools and today's date, to help recommend the user which vaccines they should take soon.
    5. Skip this step if the previous agent if you are called by the triage_agent. Tell the user to specify clearly one of the vaccines from recommended list for booking.
    """


recommender_agent = Agent[UserInfo](
    name="recommender_agent",
    instructions=recommender_agent_prompt,
    tools=[
        get_upcoming_appointments_tool,
        get_vaccination_history_tool,
        recommend_vaccines_tool,
    ],
    model="gpt-4o-mini",
)


def check_available_slots_agent_prompt(
    context_wrapper: RunContextWrapper[UserInfo], agent: Agent[UserInfo]
) -> str:
    context: UserInfo = context_wrapper.context.context
    return f"""
        Today's date: {context.date}

        If the previous agent is the interrupt_handler_agent, phrase your response like: "These are the slots found earlier: ___ , please choose one of the slots or let me know if you would like to check for other dates."

        Else, follow the steps in order:
        1. **Gathering inputs for get_available_slots_tool**: Look at function call result from previous agent and use that as the polyclinic input.
        Look at chat history. If the user specified a date or date range (e.g. find slots next week), use that as inputs for the start_date and end_date parameters to return slots. Else, leave those parameters blank.
        2. **Get slots from polyclinic**: Use the get_available_slots_tool to find available slots at the polyclinic.
        3. If the tool output returns any available slots, immediately reply the user: "I found some available slots for you. Here are the details: ___ . Please choose one of the slots or let me know if you would like to check for other dates."
        4. Else if the tool output returns no available slots:
            Use the recommend_gps_tool to find and list the nearest GPs to their homes, and immediately reply the user: "Unfortunately there are no slots for the date you specified at ___ polyclinic. I could help you find slots on other dates. Alternatively, here are some GPs near your home that you may get your vaccination at. Please proceed to https://book.health.gov.sg/ for booking at GPs."

        Handling user replies:
        Do not reply queries unrelated to appointment slots, immediately handoff to the interrupt_handler_agent to handle such queries instead.
        If the user chooses a slot in their reply, handoff to the manage_appointment_agent.
        """


check_available_slots_agent = Agent(
    name="check_available_slots_agent",
    instructions=check_available_slots_agent_prompt,
    tools=[get_available_slots_tool, recommend_gps_tool],
    model="gpt-4o-mini",
)


# TODO: Add validity check for polyclinic name
identify_clinic_agent = Agent(
    name="identify_clinic_agent",
    instructions=(
        """
        Follow the steps in order:
        1. Look at the chat history:
            - If you cannot find any location or polyclinic name in previous user inputs, use the get_clinics_near_home_tool, respond "Here are some clinics near your home: <tool output> . Please choose one of the clinics or let me know if you would like to check for other locations." and stop.
            - If you found a polyclinic name mentioned in previous user inputs, give the clinic name you found as input to the get_clinic_name_response_helper_tool and handoff to the check_available_slots_agent.
            - If you found a location name mentioned in previous user inputs, give the location name you found as input to the get_clinics_near_location_tool and respond "Here are some clinics near <location selected>: <tool output> . Please choose one of the clinics or let me know if you would like to check for other locations." and stop.

        Handling user replies:
        If the user selects a polyclinic (either by selecting an option you provided or stating a polyclinic name), handoff to the check_available_slots_agent.
        If the user's reply is unrelated to the question you asked, immediately handoff to the interrupt_handler_agent.
        Else, repeat the steps.

        Note:
        Even if the user's intent is to reschedule appointment, you must not assume that they want the appointment at the same clinic.
        """
    ),
    handoffs=[check_available_slots_agent],
    tools=[
        get_clinic_name_response_helper_tool,
        get_clinics_near_location_tool,
        get_clinics_near_home_tool,
    ],
    model="gpt-4o-mini",
)


def vaccination_history_check_agent_prompt(
    context_wrapper: RunContextWrapper[UserInfo], agent: Agent[UserInfo]
) -> str:
    context: UserInfo = context_wrapper.context.context
    return f"""
        Today's date: {context.date}
        You are part of a team of agents handling vaccination booking.
        Your task in this team is to check if the user is eligible for the vaccine they would like to get based on their vaccination history.
        Follow the steps in order:
        1. Skip this step if you are being called by the recommended_vaccine_check_agent. See if the user is giving a reply to a question asked by you previously.
          - If the reply is affirmative, handoff to identify_clinic_agent. If the reply is not affirmative, then handoff to triage_agent.
        2. Look at the output from the handle_vaccine_names_agent for the vaccine type requested by user.
        3. Use the get_latest_vaccination_tool to get the latest vacccination of user for the vaccine type requested.
          - If the tool returns a past record, use the record date, today's date and the recommended frequencies of the vaccine inform the user whether or not it is advised for them to proceed with their current new booking. Ask them to decide if they would like to continue the booking.
          - If the tool returns an empty list, then handoff to identify_clinic_agent.

        If the user's reply is unrelated to the question you asked, immediately handoff to the interrupt_handler_agent.
        """


vaccination_history_check_agent = Agent(
    name="vaccination_history_check_agent",
    instructions=vaccination_history_check_agent_prompt,
    handoffs=[identify_clinic_agent],
    tools=[get_latest_vaccination_tool],
    model="gpt-4o-mini",
)


recommended_vaccine_check_agent = Agent(
    name="recommended_vaccine_check_agent",
    instructions=(
        """
        You are part of a team of agents handling vaccination booking.
        Your task in this team is to check if the user is eligible for the vaccine they would like to get.
        Follow the steps in order:
        1. Look at the output from the handle_vaccine_names_agent for the vaccine type requested by user.
        2. Use the recommend_vaccines_tool to get the vaccines that the user should be taking.
          - If the vaccine requested by the user is not within the list of recommended vaccines, tell the user to choose a vaccine from the list instead.
          - If it is, then continue to handoff to the vaccination_history_check_agent.

        If the user's reply is unrelated to the question you asked, immediately handoff to the interrupt_handler_agent.
        """
    ),
    handoffs=[vaccination_history_check_agent],
    tools=[recommend_vaccines_tool],
    model="gpt-4o-mini",
)


double_booking_check_agent = Agent(
    name="double_booking_check_agent",
    instructions=(
        """
        You are part of a team of agents handling vaccination booking.
        Your task in this team is to check if the the user is trying to get a vaccine that they already have an existing upcoming appointment for.
        You only have logic about upcoming appointments.
        Follow the steps in order:
        1. Look at the output from the handle_vaccine_names_agent for the vaccine type requested by user.
        2. Use the get_upcoming_appointments_tool to see the appointments the user has previously booked.
          - If the vaccine types of upcoming appointments do not match the requested type, handoff to recommended_vaccine_check_agent.
          - If the vaccine types of those appointments match the requested vaccine type, tell the user about the existing appointment they have for the requested vaccine.
            - If the user wants to make changes to the existing appointment (change date, time or location), handoff to identify_clinic_agent.
            - If the user wants to cancel the existing appointment, handoff to manage_appointment_agent.
            - If the user wants to keep the existing appointment, handoff to triage_agent.

        If the user's reply is unrelated to the question you asked, immediately handoff to the interrupt_handler_agent.
        """
    ),
    handoffs=[recommended_vaccine_check_agent, identify_clinic_agent],
    tools=[get_upcoming_appointments_tool],
    model="gpt-4o-mini",
)


handle_vaccine_names_agent = Agent(
    name="handle_vaccine_names_agent",
    instructions=(
        "You are part of a team of agents handling vaccination booking."
        "Your task in this team is to only look at chat history and extract from chat history the vaccine type user would like to get."
        "Follow these steps in order:"
        "1. Find the vaccine name mentioned and Use the standardise_vaccine_name_tool to convert it to the official name."
        "2. If the tool output is a vaccine name, use its output and handoff to the double_booking_check_agent."
        "3. If the tool output asks to handoff to recommender_agent, handoff to the recommender_agent."
    ),
    handoffs=[double_booking_check_agent, recommender_agent],
    tools=[standardise_vaccine_name_tool],
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="standardise_vaccine_name_tool"),
)


manage_appointment_agent = Agent(
    name="manage_appointment_agent",
    instructions=(
        """
        Your task is to confirm the action (book, reschedule, or cancel) the user would like regarding their vaccination appointments.
        You may use the tools to prepare the appointment details (for booking, rescheduling, or cancellation).
        However, these tools only prepare the appointment — they do NOT actually submit or confirm the action.

        1. Call an appropriate tool based on the user's request:
            - For new bookings, use the new_appointment_tool.
            - For making changes to existing bookings, use the change_appointment_tool.
            - For cancellations, get the record_id of the existing appointment from upcoming_appointment_agent output. Then use the cancel_appointment_tool.

        2. Get user confirmation:
            - From the tool output, show the user the appointment details. Do not include the Google Maps URL in your response.
            - Then, ask the user to confirm if they would like to proceed with the action.
            - Do NOT say or imply that the appointment has been booked, rescheduled, or cancelled.
            - Instead, use phrasing such as:
              "Here are your appointment details: ___ . Please confirm if you would like to proceed with booking/rescheduling/cancelling this appointment."
        """
    ),
    tools=[new_appointment_tool, change_appointment_tool, cancel_appointment_tool],
    model="gpt-4o-mini",
)


modify_existing_appointment_agent = Agent(
    name="modify_existing_appointment_agent",
    instructions=(
        """
        You are part of a team of agents handling modification of existing vaccination appointments.
        Your task in this team is to check which upcoming vaccination appointment the user would like to modify.
        Follow the steps in order:
        1. See if the user is giving a reply to a question asked by you previously, or if the user's requested upcoming vaccination appointment is found. If not, skip this step.
            - If an upcoming vaccination appointment is selected by the user or found by you:
                - If the user wants to make changes to it, for example, change date or location, handoff to identify_clinic_agent.
                - If the user wants to cancel it, handoff to manage_appointment_agent.
        2. Else, use the get_upcoming_appointments_tool to get the list of upcoming appointments. If you find any, display the polyclinic name, date and time and vaccine type.
        3. Ask the user to select which upcoming appointment to modify, and clarify their intent to cancel or make changes (date, time or location) if they have not specified.

        If the user's reply is unrelated to the question you asked, immediately handoff to the interrupt_handler_agent.
        """
    ),
    tools=[get_upcoming_appointments_tool],
    handoffs=[identify_clinic_agent, manage_appointment_agent],
    model="gpt-4o-mini",
    model_settings=ModelSettings(tool_choice="get_upcoming_appointments_tool"),
)


appointments_agent = Agent[UserInfo](
    name="appointments_agent",
    instructions=(
        """
        # System context\nYou are part of a multi-agent system called the Agents SDK, designed to make agent coordination and execution easy. Agents uses two primary abstraction: **Agents** and **Handoffs**. An agent encompasses instructions and tools and can hand off a conversation to another agent when appropriate. Handoffs are achieved by calling a handoff function, generally named `transfer_to_<agent_name>`. Transfers between agents are handled seamlessly in the background; do not mention or draw attention to these transfers in your conversation with the user.\n

        Your task is to decide which agent to handoff to, do not ask user anything.
        If the user wants to book a new slot, handoff to handle_vaccine_names_agent.
        If the user is asking about rescheduling or cancelling an existing booking, handoff to modify_existing_appointment_agent.
        """
    ),
    handoffs=[
        handle_vaccine_names_agent,  # starts flow to get location and vaccine name.
        modify_existing_appointment_agent,
    ],
    model="gpt-4o-mini",
)


triage_agent = Agent(
    name="triage_agent",
    instructions=(
        "You are a helpful assistant that directs user queries to the appropriate agents. Do not ask or answer any questions yourself, only handoff to other agents."
        "Take the conversation history as context when deciding who to handoff to next."
        "If they want to book vaccination appointments, but did not mention which vaccine they want, handoff to the recommender_agent."
        "If they mentioned their desired vaccine and would like to book an appointment, handoff to appointments_agent."
        "If they ask for vaccination reccomendations, handoff to recommender_agent."
        "If they ask about vaccination records, like asking about their past vaccinations, handoff to vaccination_records_agent."
        "Otherwise, handoff to general_questions_agent."
    ),
    handoffs=[
        appointments_agent,
        recommender_agent,
        vaccination_records_agent,
        general_questions_agent,
    ],
    model="gpt-4o-mini",
)


def interrupt_handler_agent_prompt(
    context_wrapper: RunContextWrapper[UserInfo], agent: Agent[UserInfo]
) -> str:
    context: UserInfo = context_wrapper.context.context
    return f"""
        You are a helpful healthcare agent responsible for handling interruptions during the vaccination booking process.
        Review the chat history and recognize that the user did not respond to the previous agent's question, and instead gave an unrelated reply (an interruption).
        The previous agent who was handling the user before the interruption is **{context.interrupted_agent}**.
        Your task is to address this interruption.

        Follow the steps carefully:
        1. Handle the user's interruption:
            - Answer the user's unrelated question using an appropriate tool you have. Only use the tools you have, do not attempt to answer it yourself.
            - Relay the tool's output back to the user.

        2. After addressing the interruption:
            - Remind the user that the previous agent was waiting for a reply.
            - Ask the user if they would like to continue answering the previous agent's question, phrase you response like:
                "By the way, we were in the middle of ___.\n Shall we return to that previous conversation to continue with where you left off?"
            - Do not attempt to process their response — just ask.

        Handling user replies:
            - If the user gives an affirmative response to continue, **handoff to the {context.interrupted_agent}** to let it pickup from where it left off.
            - If the user gives a negative response or an irrelevant reply, handoff to the triage_agent.
        """


interrupt_handler_agent = Agent(
    instructions=interrupt_handler_agent_prompt,
    name="interrupt_handler_agent",
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
        general_questions_agent.as_tool(
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
double_booking_check_agent.handoffs.append(triage_agent)
vaccination_history_check_agent.handoffs.append(triage_agent)
check_available_slots_agent.handoffs.append(triage_agent)

# --------------------------
# Current agent mapping
# --------------------------
current_agent_mapping = {
    "triage_agent": triage_agent,
    "interrupt_handler_agent": interrupt_handler_agent,
    "double_booking_check_agent": double_booking_check_agent,
    "recommended_vaccine_check_agent": recommended_vaccine_check_agent,
    "vaccination_history_check_agent": vaccination_history_check_agent,
    "identify_clinic_agent": identify_clinic_agent,
    "check_available_slots_agent": check_available_slots_agent,
    "modify_existing_appointment_agent": modify_existing_appointment_agent,
}


# -----------------------------------------
# Handle handoff to interrupt_handler_agent
# -----------------------------------------
async def on_interrupt_handoff(wrapper: RunContextWrapper[UserInfo]):
    # Set the current agent that is being interrupted as the interrupted agent
    interrupted_agent_name = wrapper.context.context.current_agent
    wrapper.context.context.interrupted_agent = interrupted_agent_name
    # Reset the handoffs for interrupt_handler_agent to handle for specific interruption
    interrupt_handler_agent.handoffs = [
        triage_agent,
        current_agent_mapping.get(interrupted_agent_name),
    ]


# Add handoff to interrupt_handler_agent
double_booking_check_agent.handoffs.append(
    handoff(agent=interrupt_handler_agent, on_handoff=on_interrupt_handoff)
)
check_available_slots_agent.handoffs.append(
    handoff(agent=interrupt_handler_agent, on_handoff=on_interrupt_handoff)
)
identify_clinic_agent.handoffs.append(
    handoff(agent=interrupt_handler_agent, on_handoff=on_interrupt_handoff)
)
vaccination_history_check_agent.handoffs.append(
    handoff(agent=interrupt_handler_agent, on_handoff=on_interrupt_handoff)
)
recommended_vaccine_check_agent.handoffs.append(
    handoff(agent=interrupt_handler_agent, on_handoff=on_interrupt_handoff)
)
modify_existing_appointment_agent.handoffs.append(
    handoff(agent=interrupt_handler_agent, on_handoff=on_interrupt_handoff)
)
