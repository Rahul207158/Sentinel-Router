import os
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import json
from litellm import completion

# ✅ CHANGE: Import from ONNX engine
from inference_onnx import RouterEngine 

# Helper for SSE Streaming
def stream_generator(response):
    for chunk in response:
        # litellm returns ModelResponse objects
        # We need to convert them to dict, then JSON string
        chunk_dict = chunk.model_dump() # or .dict() depending on pydantic version
        json_str = json.dumps(chunk_dict)
        yield f"data: {json_str}\n\n"
    yield "data: [DONE]\n\n" 

# --- CONFIGURATION ---
# Note: Thresholds are now handled inside inference_onnx.py
# We just trust the engine's decision.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://host.docker.internal:11434") 
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "deepseek-r1:1.5b")

app = FastAPI(title="Sentinel-Router AI Gateway (ONNX)")

print("🔌 Initializing Neural Router (ONNX)...")
router = RouterEngine()

class ChatRequest(BaseModel):
    model: str = "gpt-4o"
    messages: list
    stream: bool = True

@app.post("/v1/chat/completions")
async def chat_proxy(request: ChatRequest):
    # 1. Analyze
    user_prompt = request.messages[-1]['content']
    analysis = router.predict(user_prompt)
    
    decision = analysis['decision']
    print(f"🧠 Analysis: {decision['reason']} | Risk: {decision['risk_level']}")

    # 2. Route
    local_model_id = f"ollama/{LOCAL_MODEL_NAME}"

    if decision['route'] == "BLOCK_OR_LOCAL":
        target_model = local_model_id
    elif decision['route'] == "LOCAL_DEEPSEEK":
        target_model = local_model_id
    elif decision['route'] == "CLOUD_GPT4":
        target_model = "gpt-4o"
    else:
        target_model = local_model_id # Default Fallback

    # 3. Execute
    try:
        # For Docker: We use 'host.docker.internal' to reach Ollama on your mac
        api_base = LOCAL_LLM_URL if "ollama" in target_model else None
        
        response = completion(
            model=target_model, 
            messages=request.messages,
            stream=request.stream,
            api_base=api_base,
            api_key=OPENAI_API_KEY if "gpt" in target_model else "ollama"
        )
        
        if request.stream:
            return StreamingResponse(stream_generator(response), media_type="text/event-stream")
        else:
            return response

    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)