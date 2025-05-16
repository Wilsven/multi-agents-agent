import io
import os
import time

import azure.cognitiveservices.speech as speechsdk
from azure.core.credentials import AccessToken
from azure.identity.aio import DefaultAzureCredential
from dotenv import load_dotenv

from app.schemas.voice import TranscriptionResponse

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../../../../.env"))


class SpeechToText:
    """
    This class creates a speech config object and it is created only once when the app starts.
    """

    def __init__(self):
        # Set up config variables
        self.resource_id = os.getenv("AZURE_SPEECH_SERVICE_ID")
        self.region = os.getenv("AZURE_SPEECH_SERVICE_LOCATION")
        self.credential = DefaultAzureCredential()
        self.access_token = None
        self.speech_config = None

    async def initialize(self):
        """Asynchronous initialization to set up access token and speech config."""
        self.access_token = await self.credential.get_token(
            "https://cognitiveservices.azure.com/.default"
        )
        self.speech_config = speechsdk.SpeechConfig(
            auth_token=self.get_auth_token(self.resource_id, self.access_token),
            region=self.region,
        )
        self.speech_config.set_properties(
            {
                speechsdk.properties.PropertyId.Speech_SegmentationSilenceTimeoutMs: "5000"
            }
        )

    def get_auth_token(self, resource_id, token: AccessToken):
        return "aad#" + resource_id + "#" + token.token

    async def reset_token(self):
        if self.access_token.expires_on < time.time() + 60:
            self.access_token = await self.credential.get_token(
                "https://cognitiveservices.azure.com/.default"
            )
            self.speech_config.authorization_token = self.get_auth_token(
                self.resource_id, self.access_token
            )

    async def transcribe(self, audio_file):
        await self.reset_token()

        # Read the audio file into a BytesIO stream
        audio_stream = io.BytesIO(await audio_file.read())

        stream = speechsdk.audio.PushAudioInputStream(stream_format=None)
        stream.write(audio_stream.getvalue())
        stream.close()

        audio_config = speechsdk.audio.AudioConfig(stream=stream)
        auto_detect_source_language_config = (
            speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
                languages=["en-SG", "zh-CN", "ta-IN", "ms-MY"]
            )
        )

        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config,
            audio_config=audio_config,
            auto_detect_source_language_config=auto_detect_source_language_config,
        )

        # Perform the transcription
        result = speech_recognizer.recognize_once_async().get()

        # Return the transcription result
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            response = TranscriptionResponse(text=result.text)
            return response.model_dump(), 200
        elif result.reason == result.reason == speechsdk.ResultReason.NoMatch:
            return (
                {
                    "error": "No spoken words were detected, please try saying your query again. Thank you!"
                },
                528,
            )
        else:
            return (
                {"error": f"Speech recognition error: {result.reason}"},
                500,
            )
