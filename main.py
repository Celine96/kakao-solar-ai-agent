import logging
import os

from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI, OpenAIError, APITimeoutError
import numpy as np

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

# Upstage Solar API 클라이언트 설정
client = OpenAI(
    api_key=os.getenv("UPSTAGE_API_KEY"),
    base_url="https://api.upstage.ai/v1/solar"
)

# Define Pydantic models for nested JSON structure
class DetailParams(BaseModel):
    prompt: dict

class Action(BaseModel):
    params: dict
    detailParams: dict

class RequestBody(BaseModel):
    action: Action

@app.post("/generate")
async def generate_text(request: RequestBody):
    # Extract prompt from nested JSON
    prompt = request.action.params.get("prompt")
    
    # REXA 부동산 전문가 페르소나 프롬프트
    rexa_prompt = f"""You are REXA, a chatbot that is a real estate expert with 10 years of experience in taxation (capital gains tax, property holding tax, gift/inheritance tax, acquisition tax), auctions, civil law, and building law. 
Respond politely and with a trustworthy tone, as a professional advisor would. To ensure fast responses, keep your answers under 250 tokens. 
If you don't know about the information ask the user once more time.

Question: {prompt}
And please respond in Korean following the above format."""
    
    try:
        # Call Upstage Solar API with the provided prompt
        response = client.chat.completions.create(
            model="solar-mini-nightly",  # 또는 "solar-pro", "solar-1-mini-chat"
            messages=[
                {
                    "role": "user",
                    "content": rexa_prompt
                }
            ]
        )
        # Return the generated text
        return {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": response.choices[0].message.content
                        }
                    }
                ]
            }
        }
    except APITimeoutError as e:
        logging.error(f"Upstage Solar API timeout: {e}")
        return {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": "죄송합니다. 응답 시간이 초과되었습니다. 다시 시도해주세요."
                        }
                    }
                ]
            }
        }
    except OpenAIError as e:
        logging.error(f"Upstage Solar API error: {e}")
        return {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": "죄송합니다. API 오류가 발생했습니다."
                        }
                    }
                ]
            }
        }
    except Exception as e:
        logging.error(f"Unknown error: {e}")
        return {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": "죄송합니다. 알 수 없는 오류가 발생했습니다."
                        }
                    }
                ]
            }
        }

# ==========================================
# RAG (Embeddings) - 주석처리
# ==========================================
"""
## Embeddings

import pickle

with open("embeddings.pkl", "rb") as f:
    data = pickle.load(f)
    article_chunks = data["chunks"]
    chunk_embeddings = data["embeddings"]

@app.post("/custom")
async def generate_custom(request: RequestBody):
    # Extract prompt from nested JSON
    prompt = request.action.params.get("prompt") # USER INPUT
    
    # Upstage Solar의 경우 embeddings API 엔드포인트가 다를 수 있음
    # 참고: https://console.upstage.ai/docs/capabilities/embeddings
    q_embedding = client.embeddings.create(
        input=prompt, 
        model="solar-embedding-1-large"  # Solar embedding model
    ).data[0].embedding
    
    def cosine_similarity(a, b):
        from numpy import dot
        from numpy.linalg import norm
        return dot(a, b) / (norm(a) * norm(b))

    similarities = [cosine_similarity(q_embedding, emb) for emb in chunk_embeddings]
    
    # 가장 유사한 청크 N개 선택 (여기선 2개)
    top_n = 2
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    selected_context = "\\n\\n".join([article_chunks[i] for i in top_indices])

    # Solar API에게 전달할 메시지 구성
    query = f'''Use the below context to answer the question. If the answer cannot be found, write "I don't know."

    Context:
    \"\"\"
    {selected_context}
    \"\"\"

    Question: {prompt}
    '''

    print(prompt)
    print(query)
    
    response = client.chat.completions.create(
        messages=[            
            {'role': 'user', 'content': query},
        ],
        model="solar-mini-nightly", 
        temperature=0,
    )
    
    # Return the generated text
    return {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": response.choices[0].message.content
                    }
                }
            ]
        }                
    }
"""
