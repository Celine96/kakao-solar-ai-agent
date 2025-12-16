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
import pickle

# Redis for queue management
try:
    import redis.asyncio as redis
    from redis.asyncio import Redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    Redis = Any
    logging.warning("redis package not installed. Using in-memory queue.")

# ================================================================================
# Logging Configuration
# ================================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="REXA - Real Estate Expert Assistant",
    description="Solar API + RAG chatbot for real estate",
    version="1.0.0"
)

# ================================================================================
# Configuration & Global Variables
# ================================================================================

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# Health Check Configuration
HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", 5))
MAX_UNHEALTHY_COUNT = int(os.getenv("MAX_UNHEALTHY_COUNT", 3))

# Queue Configuration
WEBHOOK_QUEUE_NAME = "rexa:webhook_queue"
WEBHOOK_PROCESSING_QUEUE = "rexa:processing_queue"
WEBHOOK_FAILED_QUEUE = "rexa:failed_queue"
MAX_RETRY_ATTEMPTS = int(os.getenv("MAX_RETRY_ATTEMPTS", 3))
QUEUE_PROCESS_INTERVAL = int(os.getenv("QUEUE_PROCESS_INTERVAL", 5))

# API Timeout Configuration
# Allowed location names (whitelist)
API_TIMEOUT = int(os.getenv("API_TIMEOUT", 4))  # 카카오톡 5초 제한 고려

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
# RAG - Load Embeddings
# ================================================================================

article_chunks = []
chunk_embeddings = []
chunk_metadata = []  # 매물 타입 메타데이터 추가

try:
    with open("embeddings.pkl", "rb") as f:
        data = pickle.load(f)
        article_chunks = data["chunks"]
        chunk_embeddings = data["embeddings"]
        # 매물 타입 메타데이터가 있으면 로드, 없으면 빈 리스트
        chunk_metadata = data.get("metadata", [])
    logger.info(f"✅ Loaded {len(article_chunks)} chunks from embeddings.pkl")
    logger.info(f"✅ RAG is ENABLED with {len(article_chunks)} chunks")
    logger.info(f"✅ Metadata loaded: {len(chunk_metadata)} entries")
except FileNotFoundError:
    logger.warning("⚠️ embeddings.pkl not found - RAG will not be available")
    logger.warning("⚠️ Server will continue WITHOUT RAG - responses will be general")
    logger.warning("⚠️ To enable RAG: run 'python embedding2_solar.py' and redeploy")
except Exception as e:
    logger.error(f"❌ Failed to load embeddings: {e}")
    logger.warning("⚠️ Server will continue WITHOUT RAG")

# ================================================================================
# 동적으로 보유 지역 추출 (메타데이터 기반)
# ================================================================================

def extract_locations_from_metadata(metadata_list):
    """메타데이터에서 보유 지역명을 추출"""
    locations = set()
    
    # 지역명 추출 패턴
    import re
    
    for meta in metadata_list:
        if meta.get("type") in ["TYPE_A", "TYPE_B"]:
            name = meta.get("name", "")
            address = meta.get("address", "")
            
            # address에서 추출 (TYPE_A)
            if address:
                # "서울특별시 강남구 학동로 401" → "강남구"
                # "서울 마포구 서교동 328-26" → "마포구", "서교동"
                dong_match = re.search(r'([가-힣]+동)', address)
                gu_match = re.search(r'([가-힣]+구)', address)
                if dong_match:
                    locations.add(dong_match.group(1))
                if gu_match:
                    locations.add(gu_match.group(1))
            
            # name에서 추출 (TYPE_B)
            # "소담빌딩 (청담동 39-7)" → "청담동"
            # "더베스트 신길동 (342-337)" → "신길동"
            dong_match = re.search(r'([가-힣]+동)', name)
            gu_match = re.search(r'([가-힣]+구)', name)
            if dong_match:
                locations.add(dong_match.group(1))
            if gu_match:
                locations.add(gu_match.group(1))
    
    return locations

# 보유 지역 자동 추출
ALLOWED_LOCATIONS = extract_locations_from_metadata(chunk_metadata)

# 서울 전체 주요 지역 (참고용)
SEOUL_ALL_LOCATIONS = {
    "강남구", "강동구", "강북구", "강서구", "관악구", "광진구", "구로구", "금천구",
    "노원구", "도봉구", "동대문구", "동작구", "마포구", "서대문구", "서초구", "성동구",
    "성북구", "송파구", "양천구", "영등포구", "용산구", "은평구", "종로구", "중구", "중랑구",
    # 주요 동
    "청담동", "논현동", "대치동", "삼성동", "역삼동", "신사동", "압구정동",
    "잠실동", "송파동", "방이동", "문정동",
    "반포동", "서초동", "방배동", "양재동",
    "신길동", "양평동", "문래동", "당산동", "여의도동",
    "신내동", "상봉동", "망우동", "중화동",
    "서교동", "연남동", "상수동", "합정동", "망원동",
    "이태원", "한남동", "용산동",
    "잠원동", "신림동", "시흥동", "봉천동",
    "성내동", "제기동", "종로", "명동", "을지로"
}

# 금지 지역 = 서울 전체 - 보유 지역
FORBIDDEN_LOCATIONS = SEOUL_ALL_LOCATIONS - ALLOWED_LOCATIONS

# 지역별 가격대 (억원 단위)
LOCATION_PRICE_RANGES = {
    "청담동": (100, 1000),    # 100억~1000억
    "논현동": (100, 1000),    # 100억~1000억  
    "대치동": (100, 2000),    # 100억~2000억
    "신내동": (5, 20),        # 5억~20억
    "상봉동": (5, 20),        # 5억~20억
    "신길동": (10, 50),       # 10억~50억
    "양평동": (5, 20),        # 5억~20억
    "문래동": (3, 10),        # 3억~10억
    "서교동": (50, 150),      # 50억~150억
    "잠원동": (10, 50),       # 10억~50억
    "종로": (5, 20),          # 5억~20억
    "신림동": (3, 10),        # 3억~10억
    "시흥동": (3, 10),        # 3억~10억
}

# 허용된 건물 용도 (보유 데이터 기반)
ALLOWED_BUILDING_TYPES = {
    "제1종근린생활시설", "제2종근린생활시설", "근린생활시설",
    "업무시설", "오피스텔", "근린상가", "상가건물", "사무실",
    "꼬마빌딩", "빌딩", "건물", "토지", "대지"
}

# 금지된 건물 용도 (보유 데이터에 없음)
FORBIDDEN_BUILDING_TYPES = {
    "다가구주택", "단독주택", "다세대주택", "아파트", "빌라",
    "연립주택", "주택", "주거용", "원룸", "투룸"
}

logger.info(f"✅ 보유 지역 자동 추출: {len(ALLOWED_LOCATIONS)}개 - {sorted(ALLOWED_LOCATIONS)}")
logger.info(f"⚠️ 금지 지역 자동 생성: {len(FORBIDDEN_LOCATIONS)}개")
logger.info(f"🔍 금지 지역 샘플: {list(sorted(FORBIDDEN_LOCATIONS))[:10]}")
logger.info(f"💰 지역별 가격대 설정: {len(LOCATION_PRICE_RANGES)}개 지역")
logger.info(f"🏢 허용 건물 용도: {len(ALLOWED_BUILDING_TYPES)}개")
logger.info(f"🚫 금지 건물 용도: {len(FORBIDDEN_BUILDING_TYPES)}개")

# ================================================================================
# RAG Helper Functions
# ================================================================================

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    from numpy import dot
    from numpy.linalg import norm
    return dot(a, b) / (norm(a) * norm(b))

async def get_relevant_context(prompt: str, top_n: int = 2) -> dict:
    """Get relevant context from embeddings for RAG
    Returns: {
        'context': str,
        'property_type': str,  # 'TYPE_A' or 'TYPE_B'
        'property_name': str
    }
    """
    if not chunk_embeddings or not article_chunks:
        logger.warning("⚠️ No embeddings available for RAG")
        return {"context": "", "property_type": "UNKNOWN", "property_name": ""}
    
    try:
        # 임베딩 차원 자동 감지
        embedding_dim = len(chunk_embeddings[0])
        logger.info(f"📊 Detected embedding dimension: {embedding_dim}")
        
        # 차원에 따라 적절한 API 사용
        if embedding_dim == 1536:
            # OpenAI 임베딩 (text-embedding-3-small)
            logger.info("🔧 Using OpenAI embedding model")
            try:
                openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                q_embedding = openai_client.embeddings.create(
                    input=prompt, 
                    model="text-embedding-3-small"
                ).data[0].embedding
            except Exception as e:
                logger.error(f"❌ OpenAI embedding failed: {e}")
                logger.info("💡 Set OPENAI_API_KEY environment variable")
                return {"context": "", "property_type": "UNKNOWN", "property_name": ""}
                
        else:
            # Solar 임베딩 (모든 다른 차원)
            logger.info(f"🔧 Using Solar embedding model (dimension: {embedding_dim})")
            try:
                q_embedding = client.embeddings.create(
                    input=prompt, 
                    model="solar-embedding-1-large-query"  # Solar 쿼리용 모델
                ).data[0].embedding
            except Exception as e:
                logger.error(f"❌ Solar embedding failed: {e}")
                logger.error(f"   Model: solar-embedding-1-large-query")
                return {"context": "", "property_type": "UNKNOWN", "property_name": ""}
        
        # Calculate similarities
        similarities = [cosine_similarity(q_embedding, emb) for emb in chunk_embeddings]
        
        # Get top N most similar chunks
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        selected_context = "\n\n".join([article_chunks[i] for i in top_indices])
        
        # 매물 타입 판단 (메타데이터가 있으면 사용, 없으면 텍스트 기반 판단)
        property_type = "TYPE_B"  # 기본값: 비제휴 중개사 매물
        property_name = ""
        
        if chunk_metadata and len(chunk_metadata) > top_indices[0]:
            # 메타데이터가 있으면 사용
            meta = chunk_metadata[top_indices[0]]
            property_type = meta.get("type", "TYPE_B")
            property_name = meta.get("name", "")
            logger.info(f"✅ Using metadata: {property_type} - {property_name}")
        else:
            # 메타데이터가 없으면 텍스트 기반 판단
            top_chunk = article_chunks[top_indices[0]]
            if "금하빌딩" in top_chunk and "서안개발" in top_chunk:
                property_type = "TYPE_A"
                property_name = "금하빌딩"
                logger.info(f"✅ Detected TYPE_A (서안개발 보유): {property_name}")
            else:
                # 매물명 추출 시도
                for line in top_chunk.split('\n'):
                    if '건물' in line or '매물' in line:
                        property_name = line.split(':')[0].strip() if ':' in line else ""
                        break
                logger.info(f"✅ Detected TYPE_B (비제휴 매물): {property_name}")
        
        # Format similarities for logging
        similarity_scores = [f"{similarities[i]:.3f}" for i in top_indices]
        logger.info(f"✅ Retrieved {top_n} relevant chunks (similarities: {similarity_scores})")
        
        return {
            "context": selected_context,
            "property_type": property_type,
            "property_name": property_name
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting relevant context: {e}")
        return {"context": "", "property_type": "UNKNOWN", "property_name": ""}

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
        logger.warning("⚠️ Using in-memory queue instead")
        use_in_memory_queue = True

async def close_redis():
    """Close Redis connection"""
    global redis_client
    if redis_client and not use_in_memory_queue:
        await redis_client.close()
        logger.info("✅ Redis connection closed")

async def enqueue_webhook_request(request_id: str, request_body: dict):
    """Enqueue a webhook request for later processing"""
    global in_memory_webhook_queue  # 전역 변수 선언
    try:
        queued_request = QueuedRequest(
            request_id=request_id,
            request_body=request_body,
            timestamp=datetime.now().isoformat(),
            retry_count=0
        )
        
        if use_in_memory_queue:
            in_memory_webhook_queue.append(queued_request)
            logger.info(f"✅ Enqueued to in-memory queue: {request_id}")
        else:
            if redis_client:
                await redis_client.lpush(
                    WEBHOOK_QUEUE_NAME,
                    queued_request.model_dump_json()
                )
                logger.info(f"✅ Enqueued to Redis: {request_id}")
    except Exception as e:
        logger.error(f"❌ Failed to enqueue request: {e}")

async def dequeue_webhook_request() -> Optional[QueuedRequest]:
    """Dequeue the next webhook request"""
    global in_memory_processing_queue, in_memory_webhook_queue  # 전역 변수 선언
    try:
        if use_in_memory_queue:
            if len(in_memory_webhook_queue) == 0:
                return None
            req = in_memory_webhook_queue.popleft()
            in_memory_processing_queue.append(req)
            return req
        
        if not redis_client:
            return None
        
        request_json = await redis_client.rpoplpush(
            WEBHOOK_QUEUE_NAME,
            WEBHOOK_PROCESSING_QUEUE
        )
        
        if not request_json:
            return None
        
        return QueuedRequest.model_validate_json(request_json)
        
    except Exception as e:
        logger.error(f"❌ Failed to dequeue request: {e}")
        return None

async def complete_webhook_request(request_id: str):
    """Mark a webhook request as completed"""
    global in_memory_processing_queue  # 전역 변수 선언 추가
    try:
        if use_in_memory_queue:
            in_memory_processing_queue = deque([
                req for req in in_memory_processing_queue 
                if req.request_id != request_id
            ])
        else:
            if redis_client:
                items = await redis_client.lrange(WEBHOOK_PROCESSING_QUEUE, 0, -1)
                for item in items:
                    req = QueuedRequest.model_validate_json(item)
                    if req.request_id == request_id:
                        await redis_client.lrem(WEBHOOK_PROCESSING_QUEUE, 1, item)
                        break
    except Exception as e:
        logger.error(f"❌ Failed to complete request: {e}")

async def fail_webhook_request(request_id: str, error_message: str):
    """Move a failed webhook request to the failed queue"""
    global in_memory_processing_queue, in_memory_failed_queue, in_memory_webhook_queue  # 전역 변수 선언
    try:
        if use_in_memory_queue:
            for req in in_memory_processing_queue:
                if req.request_id == request_id:
                    req.retry_count += 1
                    req.error_message = error_message
                    
                    if req.retry_count >= MAX_RETRY_ATTEMPTS:
                        in_memory_failed_queue.append(req)
                        in_memory_processing_queue.remove(req)
                    else:
                        in_memory_webhook_queue.appendleft(req)
                        in_memory_processing_queue.remove(req)
                    break
        else:
            if redis_client:
                items = await redis_client.lrange(WEBHOOK_PROCESSING_QUEUE, 0, -1)
                for item in items:
                    req = QueuedRequest.model_validate_json(item)
                    if req.request_id == request_id:
                        req.retry_count += 1
                        req.error_message = error_message
                        
                        await redis_client.lrem(WEBHOOK_PROCESSING_QUEUE, 1, item)
                        
                        if req.retry_count >= MAX_RETRY_ATTEMPTS:
                            await redis_client.lpush(
                                WEBHOOK_FAILED_QUEUE,
                                req.model_dump_json()
                            )
                        else:
                            await redis_client.lpush(
                                WEBHOOK_QUEUE_NAME,
                                req.model_dump_json()
                            )
                        break
    except Exception as e:
        logger.error(f"❌ Failed to fail request: {e}")

async def get_queue_sizes():
    """Get current queue sizes"""
    try:
        if use_in_memory_queue:
            return (
                len(in_memory_webhook_queue),
                len(in_memory_processing_queue),
                len(in_memory_failed_queue)
            )
        
        if not redis_client:
            return (0, 0, 0)
        
        queue_size = await redis_client.llen(WEBHOOK_QUEUE_NAME)
        processing_size = await redis_client.llen(WEBHOOK_PROCESSING_QUEUE)
        failed_size = await redis_client.llen(WEBHOOK_FAILED_QUEUE)
        
        return (queue_size, processing_size, failed_size)
        
    except Exception as e:
        logger.error(f"❌ Failed to get queue sizes: {e}")
        return (0, 0, 0)

# ================================================================================
# Background Tasks
# ================================================================================

async def health_check_monitor():
    """Monitor Solar API health"""
    global server_healthy, unhealthy_count, last_health_check
    
    logger.info("🏥 Health check monitor started")
    
    while True:
        try:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            
            # Test Solar API
            test_response = client.chat.completions.create(
                model="solar-mini",
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
                timeout=2
            )
            
            if test_response.choices[0].message.content:
                if not server_healthy:
                    logger.info("✅ Server recovered - healthy")
                server_healthy = True
                unhealthy_count = 0
            else:
                raise Exception("Empty response from API")
                
        except Exception as e:
            unhealthy_count += 1
            logger.warning(f"⚠️ Health check failed ({unhealthy_count}/{MAX_UNHEALTHY_COUNT}): {e}")
            
            if unhealthy_count >= MAX_UNHEALTHY_COUNT:
                server_healthy = False
                logger.error("❌ Server marked as unhealthy")
        
        finally:
            last_health_check = datetime.now()

async def queue_processor():
    """Process queued webhook requests"""
    logger.info("🔄 Queue processor started")
    
    while True:
        try:
            await asyncio.sleep(QUEUE_PROCESS_INTERVAL)
            
            request = await dequeue_webhook_request()
            if not request:
                continue
            
            logger.info(f"📤 Processing queued request: {request.request_id}")
            
            try:
                result = await process_solar_rag_request(request.request_body)
                await complete_webhook_request(request.request_id)
                logger.info(f"✅ Queued request {request.request_id} completed")
                
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                logger.error(f"❌ Failed to process queued request: {error_msg}")
                await fail_webhook_request(request.request_id, error_msg)
                
        except Exception as e:
            logger.error(f"❌ Queue processor error: {e}")
            await asyncio.sleep(1)

# ================================================================================
# Core Request Processing with RAG
# ================================================================================

async def process_solar_rag_request(request_body: dict):
    """Process request with Solar API + RAG"""
    
    # Extract prompt from various possible locations
    prompt = None
    
    if request_body.get("action", {}).get("params", {}).get("prompt"):
        prompt = request_body["action"]["params"]["prompt"]
        logger.info(f"✅ Method 1 (action.params.prompt): '{prompt}'")
    
    elif request_body.get("action", {}).get("detailParams", {}).get("prompt", {}).get("value"):
        prompt = request_body["action"]["detailParams"]["prompt"]["value"]
        logger.info(f"✅ Method 2 (action.detailParams.prompt.value): '{prompt}'")
    
    elif request_body.get("userRequest", {}).get("utterance"):
        prompt = request_body["userRequest"]["utterance"]
        logger.info(f"✅ Method 3 (userRequest.utterance): '{prompt}'")
    
    elif request_body.get("utterance"):
        prompt = request_body["utterance"]
        logger.info(f"✅ Method 4 (utterance): '{prompt}'")
    
    if not prompt or (isinstance(prompt, str) and prompt.strip() == ""):
        logger.warning("⚠️ No prompt found in request!")
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
    
    # 추천/리스트 요청 감지 (더 엄격하게)
    is_recommendation_request = any(keyword in prompt.lower() for keyword in 
                                    ["추천", "리스트", "목록", "몇 개", "여러 개", "3개", "5개", "10개"])
    
    # 추천 요청이면 무조건 하드코딩 목록 반환 (AI 우회)
    if is_recommendation_request:
        logger.info("🎯 Recommendation request detected - returning hardcoded list")
        return {
            "version": "2.0",
            "template": {
                "outputs": [{
                    "simpleText": {
                        "text": """현재 보유한 매물 데이터는 다음과 같습니다:

[서안개발 보유 자산]
• 금하빌딩 11층 (강남구 학동로 401) - 임대
  보증금 3.5억, 월세 2,579만원
• 서교동 328-26 (마포구 서교동) - 매매 80억
  수익률 2.43%

[시장 참고 정보 - 강남권]
• 소담빌딩 (청담동 39-7) - 약 140억원대
• 호암빌딩 (청담동 40-32) - 약 160억원대
• 청담동 39 (학동로55길 28) - 약 130억원대
• 청담동 39-13 (학동로55길 12-3) - 약 147억원대
• 남산빌 (논현동 111-31) - 약 130억원대
• 논현동 111-23 - 약 210억원대
• 논현동 62-8 - 약 500억원대
• 보성럭스타운 (논현동 254-4) - 약 500억원대
• 대치동 889-40 (선릉역 토지) - 약 1,160억원대

[시장 참고 정보 - 영등포권]
• 더베스트 신길동 (도림로 268-2) - 약 19억원대
• 문래동 카페건물 (문래북로 51-4) - 약 4.3억원대
• 양평동 또똣온반 (영등포로18길 6-1) - 약 8억원대
• 영등포 루미에르 (도림로 324-4) - 약 65억원대

[시장 참고 정보 - 중랑권]
• 신내동 신축 꼬마빌딩 (신내로10길 23) - 약 9억원대
• 상봉동 종합미싱총판 - 약 9.8억원대

[시장 참고 정보 - 기타]
• 종로 양지식 (율곡로 261) - 약 9.5억원대
• 잠원동 상가·사무실 (신반포로47길 77) - 임대상품
• 신림동 255-283 - 약 4억원대
• 시흥동 237-37 - 약 4.1억원대
• 시흥동 115-8 - 약 3.5억원대

구체적인 매물명(예: "소담빌딩", "금하빌딩 11층")을 말씀하시면 상세 정보를 안내해드립니다.

📞 상담: 서안개발 컨설팅팀 02-3443-0724"""
                    }
                }]
            }
        }
    
    # Get relevant context using RAG (단일 매물 요청만)
    rag_result = await get_relevant_context(prompt, top_n=3)  # 정확도 향상: 1->3
    context = rag_result["context"]
    property_type = rag_result["property_type"]
    property_name = rag_result["property_name"]
    
    # Build the query with context based on property type
    if context:
        # 매물 타입에 따른 프롬프트 구성
        if property_type == "TYPE_A":
            # 서안개발 보유 자산 - 직접 상담 가능
            response_guide = """응답 첫 줄: [서안개발 보유 자산] {매물명}
그 다음 줄부터 요약 형식 (bullet points):

필수 구조:
[서안개발 보유 자산] {매물명}

📍 위치: {정확한 주소}
🏢 건물: {층수, 규모}
💰 조건: {보증금/월세 또는 매매가}
✨ 특징: {주요 특징 1~2개}

📞 매매 상담: 서안개발 컨설팅팀 02-3443-0724

⚠️ 절대 규칙:
- Context에 있는 정보만 사용
- 없는 정보는 절대 지어내지 마세요
- 확실하지 않으면 "정보 없음"이라고 답변

예시:
[서안개발 보유 자산] 금하빌딩 11층

📍 위치: 서울특별시 강남구 학동로 401
🏢 건물: 지상 18층/지하 7층, 11층 143평
💰 조건: 보증금 3.5억, 월세 2,579만원
✨ 특징: 강남구청역 도보 1분, 프리미엄 오피스

📞 매매 상담: 서안개발 컨설팅팀 02-3443-0724"""
        
        else:
            # 비제휴 중개사 매물 - 시장 참고 정보로만 제공
            response_guide = """응답 첫 줄: [시장 참고 정보] {지역명} 일대 {건물명}
그 다음 줄부터 요약 형식 (bullet points):

필수 구조:
[시장 참고 정보] {지역명} 일대 {건물명/유형}

📍 위치: {구} {동} 일대
🏢 건물: {층수, 용도}
💰 시세: 약 {X}억원대 (참고가)
📐 규모: 약 {X}평대

⚠️ 마지막 줄에 반드시 포함 (필수!):
ℹ️ 본 정보는 시장 참고용이며, 정확한 내용 확인은 전문가 상담을 통해 문의해주세요

⚠️ 절대 규칙:
- Context에 있는 매물 정보만 사용
- 없는 매물은 절대 만들지 마세요
- 절대 언급 금지: 송파구, 잠실동, 반포동, 서초동, 용산구, 강서구, 강동구
- 보유 지역만 언급: 청담동, 논현동, 대치동, 신길동, 양평동, 문래동, 신내동, 상봉동, 서교동, 종로, 잠원동
- 지역별 가격대 준수:
  * 청담동/논현동/대치동: 최소 100억원 이상
  * 신내동/상봉동/양평동: 5억~20억원대
  * 신길동: 10억~50억원대
  * 문래동: 3억~10억원대
- 건물 용도 제한:
  * 허용: 제1·2종근린생활시설, 업무시설, 오피스텔, 상가건물, 꼬마빌딩
  * 금지: 다가구주택, 단독주택, 아파트, 빌라, 연립주택 (절대 사용 금지!)
- 확실한 정보만 제공
- 주소는 "○○구 ○○동 일대"만
- [시장 동향] 섹션은 사용자가 "거래 사례" 요청 시만
- 면책 문구는 절대 생략 불가!

예시:
[시장 참고 정보] 중랑구 신내동 일대 신축 꼬마빌딩

📍 위치: 중랑구 신내동 일대
🏢 건물: 2층, 제2종근린생활시설
💰 시세: 약 9억원대 (참고가)
📐 규모: 대지 99㎡, 연면적 96㎡

ℹ️ 본 정보는 시장 참고용이며, 정확한 내용 확인은 전문가 상담을 통해 문의해주세요"""

        query = f"""REXA 부동산 전문가. 요약 형식으로 간결하게.

🚨 할루시네이션 절대 금지 규칙:
1. Context에 있는 정보만 사용
2. 없는 매물은 절대 만들지 마세요
3. 절대 언급 금지 지역: 송파구, 잠실동, 반포동, 서초동, 용산구, 강서구, 강동구
4. 지역별 가격대 엄수:
   - 청담동/논현동/대치동: 최소 100억원 (20억대, 50억대 절대 금지!)
   - 신내동/양평동: 5~20억원대
   - 신길동: 10~50억원대
5. 건물 용도 제한:
   - 허용: 근린생활시설, 업무시설, 상가, 꼬마빌딩
   - 금지: 다가구주택, 아파트, 빌라 (절대 사용 금지!)
6. 확실하지 않으면 "정보 없음" 응답
7. 숫자를 지어내지 마세요

⚠️ 중요: 
1. 응답 첫 줄에 반드시 태그 표시
2. bullet points로 요약 (문장형 X)
3. 이모지 사용 (📍🏢💰📐✨📞)
4. TYPE_B는 반드시 마지막에 면책 문구 포함! (절대 생략 불가)

Type: {property_type} - {property_name}
{response_guide}

Context: {context}

질문: {prompt}

Context에 있는 사실만 사용! 송파구, 잠실동, 반포동 언급 절대 금지!
청담동/논현동은 최소 100억! 다가구주택 절대 금지!"""
        
        logger.info(f"🔍 Using RAG with {len(context)} chars of context")
        logger.info(f"🏷️ Property Type: {property_type} ({property_name})")
    
    else:
        query = f"""You are REXA, a chatbot that is a real estate expert with 10 years of experience in taxation (capital gains tax, property holding tax, gift/inheritance tax, acquisition tax), auctions, civil law, and building law. 
Respond politely and with a trustworthy tone, as a professional advisor would.

**응답 형식 가이드 (매우 중요):**
- 최대 200 토큰 이내로 간결하게 답변
- 임대조건, 건물정보 등 정보성 내용은 반드시 요약 형식으로 제공
- 불필요한 서술형 설명은 최소화하고 핵심 정보만 전달
- 숫자 정보는 명확하고 간결하게 표시


Question: {prompt}

And please respond in Korean following the above format."""
        logger.info("ℹ️ Processing without RAG context")
    
    logger.info(f"🤖 Calling Solar API with prompt: {prompt[:50]}...")
    
    try:
        response = client.chat.completions.create(
            model="solar-mini",
            messages=[{"role": "user", "content": query}],
            temperature=0.1,  # 할루시네이션 방지 (0.3 -> 0.1)
            max_tokens=500,  # 면책 문구 보장 (400 -> 500)
            timeout=API_TIMEOUT
        )
        
        answer = response.choices[0].message.content
        logger.info(f"✅ Solar API success - Response length: {len(answer)} chars")
        
        # ==================== 응답 검증 (할루시네이션 방지) ====================
        import re
        
        validation_errors = []
        
        # 1. 금지된 지역명 검증
        forbidden_found = [loc for loc in FORBIDDEN_LOCATIONS if loc in answer]
        if forbidden_found:
            validation_errors.append(f"금지 지역: {', '.join(forbidden_found)}")
        
        # 2. 금지된 건물 용도 검증
        forbidden_types = [btype for btype in FORBIDDEN_BUILDING_TYPES if btype in answer]
        if forbidden_types:
            validation_errors.append(f"금지 용도: {', '.join(forbidden_types)}")
        
        # 3. 가격대 검증 (지역별)
        price_match = re.search(r'약?\s*(\d+)억원?대', answer)
        if price_match:
            price = int(price_match.group(1))
            
            # 응답에서 지역 추출
            response_location = None
            for loc in ALLOWED_LOCATIONS:
                if loc in answer:
                    response_location = loc
                    break
            
            # 가격대 범위 확인
            if response_location and response_location in LOCATION_PRICE_RANGES:
                min_price, max_price = LOCATION_PRICE_RANGES[response_location]
                if price < min_price or price > max_price:
                    validation_errors.append(
                        f"가격 오류: {response_location}은 {min_price}~{max_price}억 범위인데 {price}억 응답"
                    )
        
        # 4. 질문-응답 지역 일치성 검증
        # 질문에서 지역 추출
        question_location = None
        for loc in ALLOWED_LOCATIONS:
            if loc in prompt:
                question_location = loc
                break
        
        # 응답에서 지역 추출
        response_location = None
        for loc in ALLOWED_LOCATIONS:
            if loc in answer:
                response_location = loc
                break
        
        # 질문 지역과 응답 지역이 다르면 오류
        if question_location and response_location:
            # "강남구청역" → "청담동" or "논현동"은 OK
            # "논현동" → "신내동"은 NG
            if question_location.endswith("동") and response_location.endswith("동"):
                if question_location != response_location:
                    validation_errors.append(
                        f"지역 불일치: 질문({question_location}) ≠ 응답({response_location})"
                    )
        
        # 검증 실패 시 에러 응답
        if validation_errors:
            logger.error(f"🚨 HALLUCINATION DETECTED: {', '.join(validation_errors)}")
            logger.error(f"   Question: {prompt}")
            logger.error(f"   Answer: {answer[:200]}")
            
            # 보유 지역 동적 생성
            dong_locations = sorted([loc for loc in ALLOWED_LOCATIONS if loc.endswith("동")])
            gu_locations = sorted([loc for loc in ALLOWED_LOCATIONS if loc.endswith("구")])
            
            return {
                "version": "2.0",
                "template": {
                    "outputs": [{
                        "simpleText": {
                            "text": f"""죄송합니다. 정확한 정보를 찾지 못했습니다.

검증 오류: {', '.join(validation_errors)}

보유 중인 매물 지역:
• 동 단위: {', '.join(dong_locations)}
• 구 단위: {', '.join(gu_locations)}

정확한 매물명(예: "소담빌딩", "금하빌딩")을 말씀하시거나,
"매물 추천" 또는 "리스트"를 요청해주세요.

📞 상담: 서안개발 컨설팅팀 02-3443-0724"""
                        }
                    }]
                }
            }
        
        logger.info(f"✅ Validation passed")
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
    return {"Hello": "REXA - Real Estate Expert Assistant (Solar + RAG)"}

@app.post("/generate")
async def generate_text(request: RequestBody):
    """REXA 부동산 전문 챗봇 with RAG - /generate 엔드포인트"""
    request_id = str(uuid.uuid4())
    
    logger.info("="*50)
    logger.info(f"📨 New request received at /generate: {request_id[:8]}")
    logger.info(f"📋 Full request body: {request.model_dump()}")
    
    try:
        # 3초 타임아웃으로 빠른 응답 시도
        result = await process_solar_rag_request(request.model_dump())
        logger.info(f"✅ Request {request_id[:8]} completed successfully")
        return result
        
    except APITimeoutError as e:
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

@app.post("/custom")
async def generate_custom(request: RequestBody):
    """REXA 부동산 전문 챗봇 with RAG - 카카오톡 5초 제한 대응"""
    request_id = str(uuid.uuid4())
    
    logger.info("="*50)
    logger.info(f"📨 New RAG request received: {request_id[:8]}")
    logger.info(f"📋 Full request body: {request.model_dump()}")
    
    try:
        # 3초 타임아웃으로 빠른 응답 시도
        result = await process_solar_rag_request(request.model_dump())
        logger.info(f"✅ Request {request_id[:8]} completed successfully")
        return result
        
    except APITimeoutError as e:
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
        mode="rexa_chatbot_rag",
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
        "timestamp": datetime.now().isoformat(),
        "rag_enabled": len(chunk_embeddings) > 0
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
        "total": queue_size + processing_size + failed_size,
        "rag_chunks_loaded": len(article_chunks)
    }

@app.post("/queue/retry-failed")
async def retry_failed_requests():
    """Manually retry all failed requests"""
    global in_memory_failed_queue, in_memory_webhook_queue  # 전역 변수 선언
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
    logger.info("="*70)
    logger.info("🚀 Starting REXA server (Solar + RAG + Property Type Detection)...")
    logger.info("="*70)
    
    # RAG 상태 확인
    if len(chunk_embeddings) > 0:
        logger.info(f"✅ RAG ENABLED: {len(chunk_embeddings)} chunks loaded")
        logger.info(f"✅ Metadata loaded: {len(chunk_metadata)} entries")
    else:
        logger.warning("⚠️ RAG DISABLED: No embeddings loaded")
        logger.warning("⚠️ Server will work but without company-specific knowledge")
    
    # Redis 초기화
    await init_redis()
    
    # Background tasks
    asyncio.create_task(health_check_monitor())
    asyncio.create_task(queue_processor())
    
    logger.info("="*70)
    logger.info("✅ REXA server startup complete!")
    logger.info(f"   - Model: solar-mini")
    logger.info(f"   - RAG chunks: {len(chunk_embeddings)}")
    logger.info(f"   - Metadata entries: {len(chunk_metadata)}")
    logger.info(f"   - Redis: {'connected' if redis_client else 'in-memory queue'}")
    logger.info("="*70)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    logger.info("👋 Shutting down REXA server (Solar + RAG)...")
    await close_redis()
    logger.info("✅ REXA server shut down successfully")
