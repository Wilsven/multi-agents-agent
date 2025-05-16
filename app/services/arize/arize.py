import os
import time
from datetime import timedelta

import pandas as pd
from phoenix import Client


class ArizeClient(Client):
    def __init__(self, project_name, user_id=None):
        # Later you can use `user_id` to fetch different credentials if needed
        api_key = os.environ.get("PHOENIX_API_KEY")
        assert (
            api_key is not None
        ), "Phoenix API key is not set in environment variables"

        # Setup the environment once
        os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={api_key}"
        os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"

        # Initialize the parent Client
        super().__init__(api_key=api_key)

        # Optional: store project name if you want to avoid hardcoding
        self.project_name = project_name

    def _filter_by_session_id(self, df, session_id):
        mask_session_id = df["attributes.session.id"] == session_id
        return df[mask_session_id]

    def _filter_by_time_range(self, df, start_time_string, end_time_string):
        start_time = pd.to_datetime(start_time_string) - timedelta(
            seconds=1
        )  # for safety
        end_time = pd.to_datetime(end_time_string) + timedelta(seconds=1)

        start_time = start_time.tz_localize("Asia/Singapore").tz_convert("UTC")
        end_time = end_time.tz_localize("Asia/Singapore").tz_convert("UTC")

        df["start_time"] = pd.to_datetime(df["start_time"])
        df["end_time"] = pd.to_datetime(df["end_time"])

        mask = (df["start_time"] >= start_time) & (df["end_time"] <= end_time)
        return df[mask]

    def get_total_tokens_usage(self, df):
        return int(df["attributes.llm.token_count.total"].sum())

    def get_agent_and_tool_count(self, df, verbose=False):
        event_lst = df["attributes.llm.output_messages"]
        num_handoff = 0
        num_tool_call = 0
        for i in range(len(event_lst)):
            msg = event_lst.iloc[i][0]
            if "message.tool_calls" in msg and msg["message.tool_calls"]:
                if msg["message.tool_calls"][0]["tool_call.function.name"].startswith(
                    "transfer"
                ):
                    num_handoff += 1
                else:
                    num_tool_call += 1

        agent_count = num_handoff + 1  # +1 for the initial triage_agent
        if verbose:
            print("Agents count:", agent_count)
            print(
                "Tools count:", num_tool_call
            )  # raw num_tool_call includes transfer/handoff functions
        return agent_count, num_tool_call

    def get_tracing_info_by_session_id(self, session_id, verbose=False):
        start_trace = time.time()

        df = self.get_spans_dataframe(project_name=self.project_name)
        filtered_df = self._filter_by_session_id(df, session_id)

        agent_count, num_tool_count = self.get_agent_and_tool_count(
            filtered_df, verbose=verbose
        )

        token_usage = self.get_total_tokens_usage(filtered_df)

        end_trace = time.time()

        if verbose:
            print("Total token usage:", token_usage)
            print("Total time taken for tracing:", end_trace - start_trace)

        return agent_count, num_tool_count, token_usage

    def get_tracing_info_in_time_interval(
        self, start_time_string, end_time_string, verbose=False
    ):

        start_trace = time.time()

        df = self.get_spans_dataframe(project_name=self.project_name)
        filtered_df = self._filter_by_time_range(df, start_time_string, end_time_string)

        agent_count, num_tool_count = self.get_agent_and_tool_count(
            filtered_df, verbose=verbose
        )
        token_usage = self.get_total_tokens_usage(filtered_df)

        end_trace = time.time()

        if verbose:
            print("Total token usage:", self.get_total_tokens_usage(filtered_df))
            print("Total time taken for tracing:", end_trace - start_trace)

        return agent_count, num_tool_count, token_usage
