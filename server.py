from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse  # This is the correct import
from fastapi.staticfiles import StaticFiles
from typing import List, Dict
import hashlib
import hmac
import json
import asyncio
from collections import deque
import logging
from datetime import datetime
import psutil  # We'll use this to get system metrics
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Meta Webhook Server")
START_TIME = time.time()

# Store for webhook events - using deque with a max size to prevent memory issues
WEBHOOK_EVENTS = deque(maxlen=100)

# Store for SSE clients
CLIENTS: List[asyncio.Queue] = []

# Replace these with your actual values from Meta
APP_SECRET = "e18fff02092b87e138b6528ccfa4a1ce"
VERIFY_TOKEN = "fitvideodemo"

@app.get("/health")
async def health_check():
    """
    Endpoint to check if the server is running and healthy.
    Returns server uptime and basic system metrics.
    """
    # Calculate how long the server has been running
    uptime_seconds = int(time.time() - START_TIME)
    
    # Get basic system metrics
    system_stats = {
        "cpu_usage": psutil.cpu_percent(),
        "memory_usage": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage('/').percent
    }
    
    # Create a comprehensive health response
    health_info = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": uptime_seconds,
        "system_metrics": system_stats
    }

    # Return a 200 OK response with health information
    return health_info

async def verify_webhook_signature(request: Request, raw_body: bytes) -> bool:
    """Verify that the webhook request came from Meta."""
    signature = request.headers.get("X-Hub-Signature-256", "")
    if not signature or not signature.startswith("sha256="):
        return False
    
    expected_signature = hmac.new(
        APP_SECRET.encode('utf-8'),
        raw_body,
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(signature[7:], expected_signature)

@app.get("/webhook")
async def verify_webhook(
    request: Request,
    hub_mode: str = None,
    hub_verify_token: str = None,
    hub_challenge: str = None
):
    """Handle the webhook verification request from Meta."""
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        logger.info("Webhook verified successfully")
        return Response(content=hub_challenge)
    raise HTTPException(status_code=403, detail="Verification failed")

@app.post("/webhook")
async def webhook(request: Request):
    """Handle incoming webhook events from Meta."""
    # Get raw body for signature verification
    raw_body = await request.body()
    
    # Verify the webhook signature
    if not await verify_webhook_signature(request, raw_body):
        raise HTTPException(status_code=403, detail="Invalid signature")
    
    # Parse the webhook payload
    try:
        payload = json.loads(raw_body)
        
        # Add timestamp to the event
        event_with_time = {
            "timestamp": datetime.now().isoformat(),
            "payload": payload
        }
        
        # Store the event
        WEBHOOK_EVENTS.append(event_with_time)
        
        # Notify all connected clients
        for client_queue in CLIENTS:
            await client_queue.put(event_with_time)
        
        logger.info(f"Received webhook: {payload}")
        return {"success": True}
    
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

async def event_generator(request: Request):
    """Generate Server-Sent Events for real-time updates."""
    client_queue = asyncio.Queue()
    CLIENTS.append(client_queue)
    
    try:
        # Send all existing events when client connects
        for event in WEBHOOK_EVENTS:
            yield f"data: {json.dumps(event)}\n\n"
        
        # Listen for new events
        while True:
            if await request.is_disconnected():
                break
            
            try:
                # Wait for new events with a timeout
                event = await asyncio.wait_for(client_queue.get(), timeout=30)
                yield f"data: {json.dumps(event)}\n\n"
            except asyncio.TimeoutError:
                # Send keepalive comment
                yield ": keepalive\n\n"
    
    finally:
        CLIENTS.remove(client_queue)

@app.get("/events")
async def events(request: Request):
    """SSE endpoint for real-time updates."""
    return EventSourceResponse(event_generator(request))

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the HTML page that displays webhook events."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Meta Webhook Events</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            #events { margin-top: 20px; }
            .event { 
                border: 1px solid #ddd;
                padding: 10px;
                margin: 10px 0;
                border-radius: 4px;
            }
            .timestamp { color: #666; font-size: 0.9em; }
            pre { white-space: pre-wrap; margin: 5px 0; }
        </style>
    </head>
    <body>
        <h1>Meta Webhook Events</h1>
        <div id="events"></div>
        <script>
            const eventsDiv = document.getElementById('events');
            
            const eventSource = new EventSource('/events');
            
            eventSource.onmessage = function(event) {
                const data = JSON.parse(event.data);
                const eventDiv = document.createElement('div');
                eventDiv.className = 'event';
                
                eventDiv.innerHTML = `
                    <div class="timestamp">${data.timestamp}</div>
                    <pre>${JSON.stringify(data.payload, null, 2)}</pre>
                `;
                
                eventsDiv.insertBefore(eventDiv, eventsDiv.firstChild);
            };
            
            eventSource.onerror = function(error) {
                console.error('EventSource failed:', error);
                eventSource.close();
                setTimeout(() => {
                    window.location.reload();
                }, 5000);
            };
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)