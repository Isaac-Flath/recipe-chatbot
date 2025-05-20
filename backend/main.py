from __future__ import annotations

"""FastAPI application entry-point for the recipe chatbot."""

from pathlib import Path
from typing import Final, List, Dict
import os
import logging

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

from backend.utils import get_agent_response  # noqa: WPS433 import from parent

# -----------------------------------------------------------------------------
# Telemetry setup
# -----------------------------------------------------------------------------

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up Phoenix telemetry
api_key = os.getenv("PHOENIX_API_KEY")  # ADD to .env file

resource = Resource.create({"service.name": "recipe-chatbot"})
tracer_provider = TracerProvider(resource=resource)

# Add console exporter for debugging
console_exporter = ConsoleSpanExporter()
tracer_provider.add_span_processor(BatchSpanProcessor(console_exporter))

# Add Phoenix exporter
exporter = OTLPSpanExporter(
    endpoint="https://app.phoenix.arize.com/v1/traces",
    headers={"api_key": api_key}
)
tracer_provider.add_span_processor(BatchSpanProcessor(exporter))

trace.set_tracer_provider(tracer_provider)
tracer = trace.get_tracer(__name__)
logger.info("Telemetry setup complete")

# -----------------------------------------------------------------------------
# Application setup
# -----------------------------------------------------------------------------

APP_TITLE: Final[str] = "Recipe Chatbot"
app = FastAPI(title=APP_TITLE)

# Serve static assets (currently just the HTML) under `/static/*`.
STATIC_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# -----------------------------------------------------------------------------
# Request / response models
# -----------------------------------------------------------------------------

class ChatMessage(BaseModel):
    """Schema for a single message in the chat history."""
    role: str = Field(..., description="Role of the message sender (system, user, or assistant).")
    content: str = Field(..., description="Content of the message.")

class ChatRequest(BaseModel):
    """Schema for incoming chat messages."""

    messages: List[ChatMessage] = Field(..., description="The entire conversation history.")


class ChatResponse(BaseModel):
    """Schema for the assistant's reply returned to the front-end."""

    messages: List[ChatMessage] = Field(..., description="The updated conversation history.")


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    """Main conversational endpoint with telemetry."""
    logger.info("Received chat request")
    with tracer.start_as_current_span("chat_endpoint") as span:
        try:
            # Add essential attributes to the span
            span.set_attribute("request.messages_count", len(payload.messages))
            if payload.messages:
                last_message = payload.messages[-1]
                span.set_attribute("request.last_message.role", last_message.role)
                span.set_attribute("request.last_message.content", last_message.content)

            # Convert Pydantic models to simple dicts for the agent
            request_messages: List[Dict[str, str]] = [msg.model_dump() for msg in payload.messages]
            logger.info(f"Processing request with {len(request_messages)} messages")
            
            updated_messages_dicts = get_agent_response(request_messages)
            logger.info("Got response from agent")
            
            # Add response attributes
            if updated_messages_dicts:
                last_response = updated_messages_dicts[-1]
                span.set_attribute("response.last_message.role", last_response["role"])
                span.set_attribute("response.last_message.content", last_response["content"])
            
            # Convert dicts back to Pydantic models for the response
            response_messages: List[ChatMessage] = [ChatMessage(**msg) for msg in updated_messages_dicts]
            return ChatResponse(messages=response_messages)

        except Exception as exc:
            logger.error(f"Error in chat endpoint: {str(exc)}")
            span.set_attribute("error.type", type(exc).__name__)
            span.set_attribute("error.message", str(exc))
            span.set_status(trace.Status(trace.StatusCode.ERROR))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:  # noqa: WPS430
    """Serve the chat UI."""

    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Frontend not found. Did you forget to build it?",
        )

    return HTMLResponse(html_path.read_text(encoding="utf-8")) 