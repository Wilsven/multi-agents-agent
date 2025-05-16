import base64
import os
import re
import time

from azure.cognitiveservices.speech import (
    ResultReason,
    SpeechConfig,
    SpeechSynthesisOutputFormat,
    SpeechSynthesisResult,
    SpeechSynthesizer,
)
from azure.core.credentials import AccessToken
from azure.identity.aio import DefaultAzureCredential


class TextToSpeech:

    def __init__(self) -> None:
        self.resource_id = os.getenv("AZURE_SPEECH_SERVICE_ID")
        self.region = os.getenv("AZURE_SPEECH_SERVICE_LOCATION")
        self.credential = DefaultAzureCredential()
        self.access_token = None
        self.speech_config = None
        self.speech_synthesizer = None

    async def initialize(self):
        """Asynchronous initialization to set up access token and speech config."""
        self.access_token = await self.credential.get_token(
            "https://cognitiveservices.azure.com/.default"
        )
        self.speech_config = SpeechConfig(
            auth_token=self.get_auth_token(self.access_token),
            region=self.region,
        )
        self.speech_config.set_speech_synthesis_output_format(
            SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
        )
        self.speech_synthesizer = SpeechSynthesizer(
            speech_config=self.speech_config, audio_config=None
        )

    def get_auth_token(self, token: AccessToken):
        return "aad#" + self.resource_id + "#" + token.token

    async def refresh_token(self):
        if self.access_token.expires_on < time.time() + 60:
            self.access_token = await self.credential.get_token(
                "https://cognitiveservices.azure.com/.default"
            )
            self.speech_config.authorization_token = self.get_auth_token(
                self.access_token
            )

    async def read_text(self, text: str) -> str:
        await self.refresh_token()

        # Force output language to be English for /voice endpoint
        self.speech_config.speech_synthesis_voice_name = "en-US-EmmaNeural"
        self.speech_synthesizer = SpeechSynthesizer(
            speech_config=self.speech_config, audio_config=None
        )

        text = re.sub(r"[*#]", "", text)  # remove * and # from markdown
        text = re.sub(
            r"-{2,}", "", text
        )  # remove 2 or more consecutive `-` from markdown
        text = re.sub(r"\[[^\]]*\]\([^\)]*\)", "", text)  # remove links from markdown
        text = re.sub(r"<br>", "", text)  # remove line breaks
        result: SpeechSynthesisResult = self.speech_synthesizer.speak_text_async(
            text
        ).get()
        if result.reason == ResultReason.SynthesizingAudioCompleted:
            return base64.b64encode(result.audio_data).decode("utf-8")
        else:
            raise Exception("Speech synthesis failed.")
