import logging
import os

from fastapi import FastAPI
from openai import OpenAI, OpenAIError, APITimeoutError
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
    try:
        # Call OpenAI API with the provided prompt
        response = client.responses.create(
            model="gpt-4.1-nano",
            input=prompt # type: ignore
        )
        # Return the generated text
        return {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": response.output_text
                        }
                    }
                ]
            }
        }
    except APITimeoutError as e:
        logging.error(f"OpenAI API timeout: {e}")
        return {"error": "OpenAI API timeout occurred."}
    except OpenAIError as e:
        logging.error(f"OpenAI API error: {e}")
        return {"error": "OpenAI API error occurred."}
    except Exception as e:
        logging.error(f"Unknown error: {e}")
        return {"error": "Unknown error occurred."}
