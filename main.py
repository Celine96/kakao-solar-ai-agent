import logging
import os
import asyncio
from datetime import datetime
from typing import Optional, Any
import uuid
from collections import deque

from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI, OpenAIError, APITimeoutError
import numpy as np

# Redis for queue management
try:
    import redis.asyncio as redis
    from redis.asyncio import Redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    Redis = Any  # Fallback type when Redis is not available
    logging.warning("redis package not installed. Using in-memory queue.")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# ================================================================================
# Configuration & Global Variables
# ================================================================================

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# Health Check Configuration
HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", 5))  # seconds
MAX_UNHEALTHY_COUNT = int(os.getenv("MAX_UNHEALTHY_COUNT", 3))

# Queue Configuration
WEBHOOK_QUEUE_NAME = "rexa:webhook_queue"
WEBHOOK_PROCESSING_QUEUE = "rexa:processing_queue"
WEBHOOK_FAILED_QUEUE = "rexa:failed_queue"
MAX_RETRY_ATTEMPTS = int(os.getenv("MAX_RETRY_ATTEMPTS", 3))
QUEUE_PROCESS_INTERVAL = int(os.getenv("QUEUE_PROCESS_INTERVAL", 5))  # seconds

# API Timeout Configuration
API_TIMEOUT = int(os.getenv("API_TIMEOUT", 3))  # seconds - 카카오톡 5초 제한 대응

# Global state
redis_client: Optional[Any] = None
server_healthy = True
unhealthy_count = 0
last_health_check = datetime.now()

# In-memory queue fallback
in_memory_webhook_queue: deque = deque()
in_memory_processing_queue: deque = deque()
in_memory_failed_queue: deque = deque()
use_in_memory_queue = False

# ================================================================================
# Upstage Solar API Configuration
# ================================================================================

client = OpenAI(
    api_key=os.getenv("UPSTAGE_API_KEY"),
    base_url="https://api.upstage.ai/v1/solar",
    timeout=API_TIMEOUT
)

logger.info("✅ Upstage Solar API client configured")

# ================================================================================
# Pydantic Models
# ================================================================================

class DetailParams(BaseModel):
    prompt: dict

class Action(BaseModel):
    params: dict
    detailParams: dict

class RequestBody(BaseModel):
    action: Action

class QueuedRequest(BaseModel):
    request_id: str
    request_body: dict
    timestamp: str
    retry_count: int = 0
    error_message: Optional[str] = None

class HealthStatus(BaseModel):
    status: str
    model: str
    mode: str
    server_healthy: bool
    last_check: str
    redis_connected: bool
    queue_size: int
    processing_queue_size: int
    failed_queue_size: int

# ================================================================================
# Redis & Queue Management
# ================================================================================

async def init_redis():
    """Initialize Redis connection"""
    global redis_client, use_in_memory_queue
    
    if not REDIS_AVAILABLE:
        logger.warning("⚠️ Redis package not installed - using in-memory queue")
        use_in_memory_queue = True
        return
    
    try:
        redis_client = await redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_keepalive=True,
            retry_on_timeout=True
        )
        await redis_client.ping()
        logger.info(f"✅ Redis connected: {REDIS_HOST}:{REDIS_PORT}")
        use_in_memory_queue = False
    except Exception as e:
        logger.warning(f"⚠️ Redis connection failed: {e}")
        logger.info("📦 Using in-memory queue as fallback")
        redis_client = None
        use_in_memory_queue = True

async def close_redis():
    """Close Redis connection"""
    global redis_client
    if redis_client:
        await redis_client.close()
        logger.info("Redis connection closed")

async def enqueue_webhook_request(request_id: str, request_body: dict) -> bool:
    """Add webhook request to queue"""
    try:
        queued_request = QueuedRequest(
            request_id=request_id,
            request_body=request_body,
            timestamp=datetime.now().isoformat(),
            retry_count=0
        )
        
        if use_in_memory_queue:
            in_memory_webhook_queue.appendleft(queued_request)
            logger.info(f"✅ Request {request_id} enqueued (in-memory)")
            return True
        
        if not redis_client:
            logger.warning("Queue not available - cannot enqueue request")
            return False
        
        await redis_client.lpush(
            WEBHOOK_QUEUE_NAME,
            queued_request.model_dump_json()
        )
        logger.info(f"✅ Request {request_id} enqueued (Redis)")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to enqueue request: {e}")
        return False

async def dequeue_webhook_request() -> Optional[QueuedRequest]:
    """Get next webhook request from queue"""
    try:
        if use_in_memory_queue:
            if len(in_memory_webhook_queue) > 0:
                request = in_memory_webhook_queue.pop()
                in_memory_processing_queue.appendleft(request)
                return request
            return None
        
        if not redis_client:
            return None
        
        result = await redis_client.brpoplpush(
            WEBHOOK_QUEUE_NAME,
            WEBHOOK_PROCESSING_QUEUE,
            timeout=1
        )
        
        if result:
            return QueuedRequest.model_validate_json(result)
        return None
    except Exception as e:
        logger.error(f"❌ Failed to dequeue request: {e}")
        return None

async def complete_webhook_request(request_id: str):
    """Mark webhook request as completed"""
    try:
        if use_in_memory_queue:
            for req in list(in_memory_processing_queue):
                if req.request_id == request_id:
                    in_memory_processing_queue.remove(req)
                    logger.info(f"✅ Request {request_id} completed (in-memory)")
                    return
            return
        
        if not redis_client:
            return
        
        processing_items = await redis_client.lrange(WEBHOOK_PROCESSING_QUEUE, 0, -1)
        for item in processing_items:
            req = QueuedRequest.model_validate_json(item)
            if req.request_id == request_id:
                await redis_client.lrem(WEBHOOK_PROCESSING_QUEUE, 1, item)
                logger.info(f"✅ Request {request_id} completed (Redis)")
                break
    except Exception as e:
        logger.error(f"❌ Failed to complete request: {e}")

async def fail_webhook_request(request_id: str, error_message: str):
    """Move failed request to failed queue or retry"""
    try:
        if use_in_memory_queue:
            for req in list(in_memory_processing_queue):
                if req.request_id == request_id:
                    req.retry_count += 1
                    req.error_message = error_message
                    in_memory_processing_queue.remove(req)
                    
                    if req.retry_count >= MAX_RETRY_ATTEMPTS:
                        in_memory_failed_queue.appendleft(req)
                        logger.error(f"❌ Request {request_id} moved to failed queue after {req.retry_count} attempts (in-memory)")
                    else:
                        in_memory_webhook_queue.appendleft(req)
                        logger.warning(f"⚠️ Request {request_id} re-queued (attempt {req.retry_count}/{MAX_RETRY_ATTEMPTS}) (in-memory)")
                    return
            return
        
        if not redis_client:
            return
        
        processing_items = await redis_client.lrange(WEBHOOK_PROCESSING_QUEUE, 0, -1)
        for item in processing_items:
            req = QueuedRequest.model_validate_json(item)
            if req.request_id == request_id:
                req.retry_count += 1
                req.error_message = error_message
                
                if req.retry_count >= MAX_RETRY_ATTEMPTS:
                    await redis_client.lpush(WEBHOOK_FAILED_QUEUE, req.model_dump_json())
                    await redis_client.lrem(WEBHOOK_PROCESSING_QUEUE, 1, item)
                    logger.error(f"❌ Request {request_id} moved to failed queue after {req.retry_count} attempts (Redis)")
                else:
                    await redis_client.lpush(WEBHOOK_QUEUE_NAME, req.model_dump_json())
                    await redis_client.lrem(WEBHOOK_PROCESSING_QUEUE, 1, item)
                    logger.warning(f"⚠️ Request {request_id} re-queued (attempt {req.retry_count}/{MAX_RETRY_ATTEMPTS}) (Redis)")
                break
    except Exception as e:
        logger.error(f"❌ Failed to handle failed request: {e}")

async def get_queue_sizes() -> tuple:
    """Get sizes of all queues"""
    try:
        if use_in_memory_queue:
            return (
                len(in_memory_webhook_queue),
                len(in_memory_processing_queue),
                len(in_memory_failed_queue)
            )
        
        if not redis_client:
            return 0, 0, 0
        
        queue_size = await redis_client.llen(WEBHOOK_QUEUE_NAME)
        processing_size = await redis_client.llen(WEBHOOK_PROCESSING_QUEUE)
        failed_size = await redis_client.llen(WEBHOOK_FAILED_QUEUE)
        return queue_size, processing_size, failed_size
    except Exception as e:
        logger.error(f"❌ Failed to get queue sizes: {e}")
        return 0, 0, 0

# ================================================================================
# Background Tasks
# ================================================================================

async def health_check_monitor():
    """Background task to monitor server health"""
    global server_healthy, unhealthy_count, last_health_check
    
    while True:
        try:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            
            is_healthy = await perform_health_check()
            
            if is_healthy:
                server_healthy = True
                unhealthy_count = 0
                logger.debug("✅ Health check passed")
            else:
                unhealthy_count += 1
                logger.warning(f"⚠️ Health check failed (count: {unhealthy_count}/{MAX_UNHEALTHY_COUNT})")
                
                if unhealthy_count >= MAX_UNHEALTHY_COUNT:
                    server_healthy = False
                    logger.error("❌ Server marked as unhealthy")
            
            last_health_check = datetime.now()
            
        except Exception as e:
            logger.error(f"❌ Health check monitor error: {e}")
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)

async def perform_health_check() -> bool:
    """Perform actual health check"""
    try:
        # Test Solar API with a simple request
        response = client.chat.completions.create(
            model="solar-mini",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=10,
            timeout=3
        )
        
        if not response or not response.choices:
            return False
        
        # Only check Redis if we're using it
        if redis_client and not use_in_memory_queue:
            await redis_client.ping()
        
        return True
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False

async def queue_processor():
    """Background task to process queued webhook requests"""
    logger.info("🚀 Queue processor started")
    
    while True:
        try:
            if not server_healthy:
                logger.warning("⏸️ Queue processing paused - server unhealthy")
                await asyncio.sleep(QUEUE_PROCESS_INTERVAL)
                continue
            
            queued_request = await dequeue_webhook_request()
            
            if queued_request:
                logger.info(f"📥 Processing queued request: {queued_request.request_id}")
                
                try:
                    result = await process_solar_request(queued_request.request_body)
                    await complete_webhook_request(queued_request.request_id)
                    logger.info(f"✅ Queued request {queued_request.request_id} processed successfully")
                    
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    logger.error(f"❌ Error processing queued request: {error_msg}")
                    await fail_webhook_request(queued_request.request_id, error_msg)
            else:
                await asyncio.sleep(QUEUE_PROCESS_INTERVAL)
                
        except Exception as e:
            logger.error(f"❌ Queue processor error: {e}")
            await asyncio.sleep(QUEUE_PROCESS_INTERVAL)

# ================================================================================
# Helper Functions
# ================================================================================

async def process_solar_request(request_body: dict) -> dict:
    """Process Solar API request with comprehensive parameter extraction"""
    
    # 상세 로그: 모든 요청 기록
    logger.info("="*70)
    logger.info("🔍 PARAMETER EXTRACTION START")
    logger.info(f"📋 Full request body: {request_body}")
    
    # 다양한 방법으로 prompt 추출 시도
    prompt = None
    
    # 방법 1: action.params.prompt (표준)
    if request_body.get("action", {}).get("params", {}).get("prompt"):
        prompt = request_body["action"]["params"]["prompt"]
        logger.info(f"✅ Method 1 (action.params.prompt): '{prompt}'")
    
    # 방법 2: action.detailParams
    elif request_body.get("action", {}).get("detailParams", {}):
        detail_params = request_body["action"]["detailParams"]
        for key, value in detail_params.items():
            if isinstance(value, dict) and "value" in value:
                prompt = value["value"]
                logger.info(f"✅ Method 2 (detailParams.{key}): '{prompt}'")
                break
    
    # 방법 3: userRequest.utterance (카카오톡 직접 발화)
    elif request_body.get("userRequest", {}).get("utterance"):
        prompt = request_body["userRequest"]["utterance"]
        logger.info(f"✅ Method 3 (userRequest.utterance): '{prompt}'")
    
    # 방법 4: 최상위 utterance
    elif request_body.get("utterance"):
        prompt = request_body["utterance"]
        logger.info(f"✅ Method 4 (utterance): '{prompt}'")
    
    # prompt를 찾지 못한 경우
    if not prompt or (isinstance(prompt, str) and prompt.strip() == ""):
        logger.warning("⚠️ No prompt found in request!")
        logger.warning(f"⚠️ Request keys: {list(request_body.keys())}")
        
        # 기본 응답
        return {
            "version": "2.0",
            "template": {
                "outputs": [{
                    "simpleText": {
                        "text": "안녕하세요! REXA입니다. 무엇이 궁금하신가요?\n부동산 세금, 경매, 민법 등에 대해 질문해주세요."
                    }
                }]
            }
        }
    
    logger.info(f"📝 Final extracted prompt: '{prompt}'")
    logger.info("="*70)
    
    rexa_prompt = f"""You are REXA, a chatbot that is a real estate expert with 10 years of experience in taxation (capital gains tax, property holding tax, gift/inheritance tax, acquisition tax), auctions, civil law, and building law. 
Respond politely and with a trustworthy tone, as a professional advisor would. To ensure fast responses, keep your answers under 250 tokens. 
If you don't know about the information ask the user once more time.

Question: {prompt}
And please respond in Korean following the above format."""
    
    logger.info(f"🤖 Calling Solar API with prompt: {prompt[:50]}...")
    
    try:
        response = client.chat.completions.create(
            model="solar-mini",
            messages=[{"role": "user", "content": rexa_prompt}],
            timeout=API_TIMEOUT
        )
        
        answer = response.choices[0].message.content
        logger.info(f"✅ Solar API success - Response length: {len(answer)} chars")
        logger.info(f"📤 Sending response: {answer[:100]}...")
        
        return {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": answer
                        }
                    }
                ]
            }
        }
        
    except APITimeoutError as e:
        logger.error(f"⏰ API Timeout after {API_TIMEOUT}s: {e}")
        raise
    except OpenAIError as e:
        logger.error(f"❌ OpenAI API Error: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error: {type(e).__name__}: {e}")
        raise

# ================================================================================
# API Endpoints
# ================================================================================

@app.get("/")
def read_root():
    return {"Hello": "REXA - Real Estate Expert Assistant (Solar)"}

@app.post("/generate")
async def generate_text(request: RequestBody):
    """REXA 부동산 전문 챗봇 - 카카오톡 5초 제한 대응"""
    request_id = str(uuid.uuid4())
    
    # 상세 로그: 모든 요청 기록
    logger.info("="*50)
    logger.info(f"📨 New request received: {request_id[:8]}")
    logger.info(f"📋 Full request body: {request.model_dump()}")
    
    try:
        # 3초 타임아웃으로 빠른 응답 시도
        result = await process_solar_request(request.model_dump())
        logger.info(f"✅ Request {request_id[:8]} completed successfully")
        return result
        
    except APITimeoutError as e:
        # 3초 타임아웃 발생 시
        logger.warning(f"⏰ Timeout (3s) - enqueueing request {request_id}")
        await enqueue_webhook_request(request_id, request.model_dump())
        
        return {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": "답변 생성에 시간이 걸리고 있습니다. 잠시 후 다시 질문해주세요."
                        }
                    }
                ]
            }
        }
        
    except OpenAIError as e:
        logger.error(f"❌ API Error: {e}")
        await enqueue_webhook_request(request_id, request.model_dump())
        
        return {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": "일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
                        }
                    }
                ]
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error: {type(e).__name__}: {e}")
        await enqueue_webhook_request(request_id, request.model_dump())
        
        return {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": "죄송합니다. 오류가 발생했습니다. 다시 한번 질문해주시겠어요?"
                        }
                    }
                ]
            }
        }

@app.get("/health")
async def health_check() -> HealthStatus:
    """Enhanced health check endpoint"""
    queue_size, processing_size, failed_size = await get_queue_sizes()
    
    return HealthStatus(
        status="healthy" if server_healthy else "unhealthy",
        model="solar-mini",
        mode="rexa_chatbot",
        server_healthy=server_healthy,
        last_check=last_health_check.isoformat(),
        redis_connected=(redis_client is not None and not use_in_memory_queue),
        queue_size=queue_size,
        processing_queue_size=processing_size,
        failed_queue_size=failed_size
    )

@app.get("/health/ping")
async def health_ping():
    """Simple ping endpoint for client health checks"""
    return {
        "alive": True,
        "healthy": server_healthy,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/queue/status")
async def queue_status():
    """Get detailed queue status"""
    queue_size, processing_size, failed_size = await get_queue_sizes()
    
    return {
        "queue_type": "in-memory" if use_in_memory_queue else "redis",
        "webhook_queue": queue_size,
        "processing_queue": processing_size,
        "failed_queue": failed_size,
        "total": queue_size + processing_size + failed_size
    }

@app.post("/queue/retry-failed")
async def retry_failed_requests():
    """Manually retry all failed requests"""
    try:
        if use_in_memory_queue:
            retry_count = len(in_memory_failed_queue)
            while len(in_memory_failed_queue) > 0:
                req = in_memory_failed_queue.pop()
                req.retry_count = 0
                in_memory_webhook_queue.appendleft(req)
            
            logger.info(f"✅ Retrying {retry_count} failed requests (in-memory)")
            return {"retried": retry_count, "queue_type": "in-memory"}
        
        if not redis_client:
            return {"error": "Queue not available"}
        
        failed_items = await redis_client.lrange(WEBHOOK_FAILED_QUEUE, 0, -1)
        retry_count = 0
        
        for item in failed_items:
            req = QueuedRequest.model_validate_json(item)
            req.retry_count = 0
            await redis_client.lpush(WEBHOOK_QUEUE_NAME, req.model_dump_json())
            retry_count += 1
        
        await redis_client.delete(WEBHOOK_FAILED_QUEUE)
        
        logger.info(f"✅ Retrying {retry_count} failed requests (Redis)")
        return {"retried": retry_count, "queue_type": "redis"}
        
    except Exception as e:
        logger.error(f"❌ Failed to retry requests: {e}")
        return {"error": str(e)}

# ================================================================================
# Startup & Shutdown Events
# ================================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    logger.info("🚀 Starting REXA server (Solar)...")
    
    await init_redis()
    
    asyncio.create_task(health_check_monitor())
    asyncio.create_task(queue_processor())
    
    logger.info("✅ REXA server (Solar) started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    logger.info("👋 Shutting down REXA server (Solar)...")
    await close_redis()
    logger.info("✅ REXA server (Solar) shut down successfully")
