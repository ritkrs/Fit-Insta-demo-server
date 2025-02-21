from fastapi import FastAPI, Request, Response, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from typing import List
import hashlib
import hmac
import json
import asyncio
from collections import deque
import logging
from datetime import datetime, timedelta
import psutil
import time
import os
from dotenv import load_dotenv
import requests
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

from celery import Celery
import random

# Download required data (only once)
nltk.download('vader_lexicon')

load_dotenv()

# Configure logging

# Initialize FastAPI app
app = FastAPI(title="Meta Webhook Server")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()


# Store webhook events - using deque with max size to prevent memory issues
WEBHOOK_EVENTS = deque(maxlen=100)

# Store SSE clients
CLIENTS: List[asyncio.Queue] = []

# Webhook Credentials
APP_SECRET = os.getenv("APP_SECRET", "e18fff02092b87e138b6528ccfa4a1ce")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "fitvideodemo")
access_token = "IGAAI8SJHk0mNBZAFB6TF9zejQtcnoyWWlOaGRSaEJyRGlfTXVUMEdveGJiVURXRXNlOUUwZA0QwQ2w4ZAi1HVE5mM2tqdk1jYW94VHVQbHdnWUx1NVduTHg1QzRMY1BzMVdqaEpId3B3X0JxNzM4dWJmWGtsWnZAKb1p4SnNiRzFMZAwZDZD"  # Replace with your actual token
account_id = "17841472117168408"  # Replace
openrouter_mistaralnemo_api_key = "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"

default_dm_response_positive = "Thanks for the message, we appreciate it!"
default_dm_response_negative = "We apologize for any mistakes on our part. Please reach out to us at mail_id@email.com for further assistance."
default_comment_response_positive = "Thanks for the message, we appreciate it!"
default_comment_response_negative = "We apologize for any mistakes on our part. Please reach out to us at mail_id@email.com for further assistance."
# Save Webhook Events to JSON File
WEBHOOK_FILE = "webhook_events.json"


# --- Celery Setup ---
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL")  # Use environment variable
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND")  # and sensible defaults

celery = Celery(__name__, broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)
celery.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',  # Set a consistent timezone
    enable_utc=True,
)


@celery.task(name="send_delayed_dm")
def send_delayed_dm(access_token, recipient_id, message_to_be_sent):
    """Sends a delayed direct message."""
    try:
        result = postmsg(access_token, recipient_id, message_to_be_sent)
        logger.info(f"DM sent to {recipient_id}.  Result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error sending DM to {recipient_id}: {e}")
        raise  # Re-raise the exception for Celery to handle retries (if configured)


@celery.task(name="send_delayed_reply")
def send_delayed_reply(access_token, comment_id, message_to_be_sent):
    """Sends a delayed reply to a comment."""
    try:
        result = sendreply(access_token, comment_id, message_to_be_sent)
        logger.info(f"Reply sent to comment {comment_id}. Result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error sending reply to comment {comment_id}: {e}")
        raise  # Important: Re-raise for Celery retry handling.



def save_events_to_file():
    """Save webhook events to a JSON file."""
    with open(WEBHOOK_FILE, "w") as f:
        json.dump(list(WEBHOOK_EVENTS), f, indent=4)


def load_events_from_file():
    """Load webhook events from the JSON file (if it exists)."""
    if os.path.exists(WEBHOOK_FILE):
        try:
            with open(WEBHOOK_FILE, "r") as f:
                events = json.load(f)
                WEBHOOK_EVENTS.extend(events)
        except Exception as e:
            logger.error(f"Failed to load events from file: {e}")

def llm_response(text):
    with open("system_prompt.txt", "r") as file:
        system_prompt = file.read().strip()

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": "Bearer " + openrouter_mistaralnemo_api_key,
            "Content-Type": "application/json",
            "HTTP-Referer": "",  # Optional. Site URL for rankings on openrouter.ai.
            "X-Title": "",  # Optional. Site title for rankings on openrouter.ai.
        },
        data=json.dumps({
            "model": "mistralai/mistral-nemo:free",
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
        })
    )
    return response.choices[0].message.content

def postmsg(access_token, recipient_id, message_to_be_sent):
    url = "https://graph.instagram.com/v21.0/me/messages"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    json_body = {
        "recipient": {
            "id": recipient_id
        },
        "message": {
            "text": message_to_be_sent
        }
    }

    response = requests.post(url, headers=headers, json=json_body)
    data = response.json()
    return data

def sendreply(access_token, comment_id, message_to_be_sent):
    url = f"https://graph.instagram.com/v22.0/{comment_id}/replies"

    params = {
        "message": message_to_be_sent,
        "access_token": access_token
    }

    response = requests.post(url, params=params)
    data = response.json()
    return data


def parse_instagram_webhook(data):
    """
    Parse Instagram webhook events for both direct messages and comments.

    Args:
        data (dict): The full webhook payload received from Meta

    Returns:
        list: A list of parsed event dictionaries
    """
    results = []

    try:
        # Extract timestamp from the wrapper data
        event_timestamp = data.get("timestamp")

        # Handle different possible payload structures
        payload = data.get("payload", data) if isinstance(data, dict) else data

        # Extract entries from payload
        entries = payload.get("entry", [])

        logger.info(f"Number of entries found: {len(entries)}")

        for entry in entries:
            # Process Direct Messages
            messaging_events = entry.get("messaging", [])
            for messaging_event in messaging_events:
                sender = messaging_event.get("sender", {})
                recipient = messaging_event.get("recipient", {})
                message = messaging_event.get("message", {})

                if message:
                    message_event_details = {
                        "type": "direct_message",
                        "sender_id": sender.get("id"),
                        "recipient_id": recipient.get("id"),
                        "text": message.get("text"),
                        "message_id": message.get("mid"),
                        "timestamp": event_timestamp,
                        "entry_time": entry.get("time"),
                        "is_echo": message.get("is_echo", False)
                    }
                    results.append(message_event_details)

            # Process Comments
            changes = entry.get("changes", [])
            for change in changes:
                if change.get("field") == "comments":
                    comment_value = change.get("value", {})
                    if comment_value:
                        comment_details = {
                            "type": "comment",
                            "comment_id": comment_value.get("id"),
                            "text": comment_value.get("text"),
                            "timestamp": event_timestamp,
                            "media_id": comment_value.get("media", {}).get("id"),
                            "media_type": comment_value.get("media", {}).get("media_product_type"),
                            "from_username": comment_value.get("from", {}).get("username"),
                            "from_id": comment_value.get("from", {}).get("id"),
                            "entry_time": entry.get("time")
                        }
                        results.append(comment_details)

    except Exception as e:
        logger.error(f"Parsing error: {e}")
        logger.error(f"Problematic payload: {json.dumps(data, indent=2)}")

    return results

def analyze_sentiment(comment_text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(comment_text)

    # Determine sentiment based on compound score
    if sentiment_scores['compound'] > 0.25:
        sentiment = "Positive"
    else:
        sentiment = "Negative"  # Neutral sentiment will return None

    return sentiment

# Load events from file on startup
load_events_from_file()

@app.get("/ping")
def ping():
    return {"message": "Server is active"}

@app.get("/health")
async def health_check():
    """Check server health status."""
    uptime_seconds = int(time.time() - START_TIME)
    system_stats = {
        "cpu_usage": psutil.cpu_percent(),
        "memory_usage": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent
    }
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": uptime_seconds,
        "system_metrics": system_stats
    }


async def verify_webhook_signature(request: Request, raw_body: bytes) -> bool:
    """Verify that the webhook request is from Meta."""
    signature = request.headers.get("X-Hub-Signature-256", "")
    if not signature or not signature.startswith("sha256="):
        logger.error("Signature is missing or not properly formatted")
        return False

    expected_signature = hmac.new(
        APP_SECRET.encode('utf-8'),
        raw_body,
        hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(signature[7:], expected_signature):
        logger.error(f"Signature mismatch: {signature[7:]} != {expected_signature}")
        return False

    return True


@app.get("/webhook")
async def verify_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge: str = Query(None, alias="hub.challenge")
):
    """Verify webhook from Meta."""
    logger.info(f"Received verification request: {hub_mode}, {hub_verify_token}")

    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        logger.info("Webhook verification successful")
        return Response(content=hub_challenge, media_type="text/plain")

    logger.error("Webhook verification failed")
    raise HTTPException(status_code=403, detail="Verification failed")

message_queue = {}
@celery.task(name="send_dm")
def send_dm(message_queue):
    try:
        for conversation_id, messages in message_queue.items():
            if not messages:
                continue

            # Get recipient ID from the first message
            recipient_id = messages[0]["sender_id"]

            # Combine all messages from the same conversation
            combined_text = "\n".join([msg["text"] for msg in messages])

            # Generate response using LLM (assuming llm_response function exists)
            try:
                response = llm_response(combined_text)
            except Exception as e:
                logger.error(f"Error generating LLM response: {e}")
                response = default_dm_response_positive  # Fallback response

            # Send the combined response
            try:
                result = postmsg(access_token, recipient_id, response)
                logger.info(f"Sent combined response to {recipient_id}. Result: {result}")
            except Exception as e:
                logger.error(f"Error sending message to {recipient_id}: {e}")

        return {"status": "success", "processed_conversations": len(message_queue)}

    except Exception as e:
        logger.error(f"Error in send_dm task: {e}")
        raise
    finally:
        # Clear the message queue after processing
        message_queue.clear()


@app.post("/webhook")
async def webhook(request: Request):
    """Handle incoming webhook events from Meta."""
    raw_body = await request.body()
    logger.info(f"Received raw webhook payload: {raw_body.decode('utf-8')}")

    if not await verify_webhook_signature(request, raw_body):
        raise HTTPException(status_code=403, detail="Invalid signature")

    try:
        payload = json.loads(raw_body)
        event_with_time = {
            "timestamp": datetime.now().isoformat(),
            "payload": payload
        }

        # Parse the webhook and get events
        parsed_events = parse_instagram_webhook(event_with_time)
        logger.info("Parsed Webhook Events:")
        for event in parsed_events:
            logger.info(json.dumps(event, indent=2))

            # Handle different types of events
            if event["type"] == "direct_message" and event["recipient_id"] == account_id and event["is_echo"] == False:
                # Analyze sentiment of the message
#                sentiment = analyze_sentiment(event["text"])
#                if sentiment == "Positive":
#                    message_to_be_sent = default_dm_response_positive
#                else:
#                    message_to_be_sent = default_dm_response_negative
#
#                # Schedule the DM task
#                delay = random.randint(1 * 60, 2 * 60)  # 10 to 25 minutes in seconds
#                send_delayed_dm.apply_async(
#                    args=(access_token, event["sender_id"], message_to_be_sent),
#                    countdown=delay, expires=delay+60
#                )
                delay = random.randint(1 * 60, 2 * 60)  # 10 to 25 minutes in seconds
                conversation_id = str(event["sender_id"]) + "_" + str(event["recipient_id"]) # Corrected variable name

                if conversation_id not in message_queue:
                    message_queue[conversation_id] = []
                message_queue[conversation_id].append(event)

                if len(message_queue[conversation_id]) == 1 : # trigger only when first message comes
                    send_dm.apply_async(args = (message_queue,), countdown = delay, expires=delay+60)
                    logger.info(f"Scheduled DM task for sender {event['sender_id']} in {delay} seconds")


            elif event["type"] == "comment" and event["from_id"] != account_id:
                # Analyze sentiment of the comment
                sentiment = analyze_sentiment(event["text"])
                if sentiment == "Positive":
                    message_to_be_sent = default_comment_response_positive
                else:
                    message_to_be_sent = default_comment_response_negative

                # Schedule the reply task
                delay = random.randint(1 * 60, 2 * 60)  # 10 to 25 minutes in seconds
                send_delayed_reply.apply_async(
                    args=(access_token, event["comment_id"], message_to_be_sent),
                    countdown=delay,expires=delay+60
                )
                logger.info(f"Scheduled reply task for comment {event['comment_id']} in {delay} seconds")


        # Store event and notify clients
        WEBHOOK_EVENTS.append(event_with_time)
        save_events_to_file()

        # Notify connected SSE clients
        for client_queue in CLIENTS:
            await client_queue.put(event_with_time)

        return {"success": True, "parsed_events": parsed_events}

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

@app.get("/webhook_events")
async def get_webhook_events():
    """Retrieve all stored webhook events."""
    return {"events": list(WEBHOOK_EVENTS)}


async def event_generator(request: Request):
    """Generate Server-Sent Events."""
    client_queue = asyncio.Queue()
    CLIENTS.append(client_queue)

    try:
        # Send existing events
        for event in WEBHOOK_EVENTS:
            yield f"data: {json.dumps(event)}\n\n"

        # Listen for new events
        while not await request.is_disconnected():
            try:
                event = await asyncio.wait_for(client_queue.get(), timeout=30)
                yield f"data: {json.dumps(event)}\n\n"
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"

    finally:
        CLIENTS.remove(client_queue)


@app.get("/events")
async def events(request: Request):
    """SSE endpoint for real-time webhook events."""
    return EventSourceResponse(event_generator(request))


# Serve static HTML
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)