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
from datetime import datetime
import psutil
import time
import os
from dotenv import load_dotenv
import requests
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

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
access_token = "IGAAI8SJHk0mNBZAFB6TF9zejQtcnoyWWlOaGRSaEJyRGlfTXVUMEdveGJiVURXRXNlOUUwZA0QwQ2w4ZAi1HVE5mM2tqdk1jYW94VHVQbHdnWUx1NVduTHg1QzRMY1BzMVdqaEpId3B3X0JxNzM4dWJmWGtsWnZAKb1p4SnNiRzFMZAwZDZD"
account_id = "17841472117168408"

default_dm_response_positive = "Thanks for the message, we appreciate it!"
default_dm_response_negative = "We apologize for any mistakes on our part. Please reach out to us at mail_id@email.com for further assistance."
default_comment_response_positive = "Thanks for the message, we appreciate it!"
default_comment_response_negative = "We apologize for any mistakes on our part. Please reach out to us at mail_id@email.com for further assistance."
# Save Webhook Events to JSON File
WEBHOOK_FILE = "webhook_events.json"


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
    comment_id = "18089022370492854"  
    access_token = "IGAAI8SJHk0mNBZAFB6TF9zejQtcnoyWWlOaGRSaEJyRGlfTXVUMEdveGJiVURXRXNlOUUwZA0QwQ2w4ZAi1HVE5mM2tqdk1jYW94VHVQbHdnWUx1NVduTHg1QzRMY1BzMVdqaEpId3B3X0JxNzM4dWJmWGtsWnZAKb1p4SnNiRzFMZAwZDZD"

    url = f"https://graph.instagram.com/v22.0/{comment_id}/replies"

    params = {
        "message": message_to_be_sent,
        "access_token": access_token
    }

    response = requests.post(url, params=params)
    data = response.json()
    return data

def handle_comment(access_token: str, comment_data: dict):
    """
    Handle Instagram comment interactions.
    """
    try:
        comment_text = comment_data.get('text', '')
        sentiment = analyze_sentiment(comment_text)
        
        # You'll need to implement this function similar to postmsg
        # This is a placeholder for where you'd put your comment reply logic
        if sentiment == "Positive":
            # Add your comment reply logic here
            logger.info(f"Positive comment received: {comment_text}")
            sendreply(access_token, comment_data['comment_id'], default_comment_response_positive)
            logger.info(f"Comment reply sent: {default_comment_response_positive}")
        
        else:
            logger.info(f"Negative comment received: {comment_text}")
            sendreply(access_token, comment_data['comment_id'], default_comment_response_negative)
            logger.info(f"Comment reply sent: {default_comment_response_negative}")

    except Exception as e:
        logger.error(f"Error handling comment: {e}")

# Function to handle DM responses (modified version of your existing logic)
def handle_dm(access_token: str, message_data: dict):
    """
    Handle Instagram direct message interactions.
    """
    try:
        if not message_data.get("is_echo", False):  # Only respond to non-echo messages
            sender_id = message_data.get("sender_id")
            text_message = message_data.get("text", "")
            
            if analyze_sentiment(text_message) == "Positive":
                logger.info(f"Positive DM received: {text_message}")
                postmsg(access_token, sender_id, default_dm_response_positive)
                logger.info(f"DM reply sent: {default_dm_response_positive}")
            else:
                logger.info(f"Negative DM received: {text_message}")
                postmsg(access_token, sender_id, default_dm_response_negative)
                logger.info(f"DM reply sent: {default_dm_response_negative}")
                
    except Exception as e:
        logger.error(f"Error handling DM: {e}")

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
    if sentiment_scores['compound'] > 0.05:
        sentiment = "Positive"
    else:
        sentiment = "Negative"  # Neutral sentiment will return None

    return sentiment

# Load events from file on startup
load_events_from_file()


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
            if event["type"] == "direct_message":
                handle_dm(access_token, event)
            elif event["type"] == "comment":
                handle_comment(access_token, event)

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
