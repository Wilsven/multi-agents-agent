import asyncio
import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from openinference.instrumentation.openai import OpenAIInstrumentor
from phoenix.otel import register
from starlette.middleware.cors import CORSMiddleware
from app.routers import (
    chat,
    metrics,
    voice,
)
from app.services.arize.arize import ArizeClient
from app.services.speech.speech_to_text import SpeechToText
from app.services.speech.text_to_speech import TextToSpeech

logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

def create_agent_app():
    
    @asynccontextmanager
    async def agent_lifespan(app: FastAPI):
        # initialize Text-to-Speech service
        tts_service = TextToSpeech()
        await tts_service.initialize()
        app.state.text_to_speech_service = tts_service

        # Add Phoenix API Key for tracing
        PHOENIX_API_KEY = os.environ.get("PHOENIX_API_KEY")
        os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
        os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"

        # configure the Phoenix tracer
        project_name = "my-llm-app-test"
        tracer_provider = register(
            project_name=project_name,
            verbose=False,
        )
        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

        # Init Arize metrics getter
        app.state.arize_getter = ArizeClient(project_name)

        yield

    app = FastAPI(lifespan=agent_lifespan)

    # origins = [
    #     "http://localhost:3000",
    #     "http://localhost:4200",
    #     "http://localhost:8000",
    # ]
    origins = ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat.router)
    app.include_router(voice.router)
    app.include_router(metrics.router)

    @app.get("/")
    async def agent_root():
        return JSONResponse(content={"detail": "Hello Agent!"})

    return app

agent_app = create_agent_app()


async def run_apps():
    config_agent = uvicorn.Config(
        "main:agent_app", host="127.0.0.1", port=8001, reload=True, reload_dirs=["."]
    )
    server_agent = uvicorn.Server(config_agent)

    # Run both servers concurrently
    await asyncio.gather(server_agent.serve())


# main function to run the app
if __name__ == "__main__":
    asyncio.run(run_apps())
