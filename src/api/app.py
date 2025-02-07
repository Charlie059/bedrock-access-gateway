import logging
import boto3
from typing import List, Dict

import uvicorn
from fastapi import FastAPI, Request, Response, Header, HTTPException, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse, StreamingResponse
from mangum import Mangum
import json
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uuid
import time
import re

from api.setting import API_ROUTE_PREFIX, TITLE, DESCRIPTION, SUMMARY, VERSION

config = {
    "title": TITLE,
    "description": DESCRIPTION,
    "summary": SUMMARY,
    "version": VERSION,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
app = FastAPI(**config)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    allow_origin_regex="http://.*"
)

# 注释掉这些行
# from api.routers import model, chat, embeddings

# app.include_router(model.router, prefix=API_ROUTE_PREFIX)
# app.include_router(chat.router, prefix=API_ROUTE_PREFIX)
# app.include_router(embeddings.router, prefix=API_ROUTE_PREFIX)

# 添加安全验证
security = HTTPBearer()

# 修改 boto3 客户端
bedrock_runtime = boto3.client('bedrock-runtime')
bedrock = boto3.client('bedrock')
sagemaker = boto3.client('sagemaker')
sagemaker_runtime = boto3.client('sagemaker-runtime')

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """验证 Bearer token"""
    return credentials.credentials

@app.get("/health")
async def health():
    """For health check if needed"""
    return {"status": "OK"}


@app.get("/api/v1")
async def api_root():
    """OpenAI API 根路径"""
    return JSONResponse({
        "data": None,
        "object": "list"
    })


async def get_bedrock_models() -> List[Dict]:
    """获取 Bedrock 可用模型列表"""
    try:
        response = bedrock.list_foundation_models()
        models = []
        for model in response.get('modelSummaries', []):
            models.append({
                "id": model['modelId'],
                "object": "model",
                "created": 1738887636,
                "owned_by": "bedrock"
            })
        return models
    except Exception as e:
        logging.error(f"Error getting Bedrock models: {e}")
        return []

async def get_sagemaker_models() -> List[Dict]:
    """获取 SageMaker 端点列表"""
    return [{
        "id": "arn:aws:sagemaker:us-east-1:623586450996:endpoint/r1-8b-endpoint",
        "object": "model",
        "created": 1706745202,
        "owned_by": "sagemaker"
    }]

@app.get("/api/v1/models")
async def list_models(token: str = Depends(verify_token)):
    """返回可用模型列表"""
    # 获取两种模型列表
    bedrock_models = await get_bedrock_models()
    sagemaker_models = await get_sagemaker_models()
    
    # 合并列表
    all_models = bedrock_models + sagemaker_models
    
    return {
        "object": "list",
        "data": all_models
    }


@app.post("/api/v1/chat/completions")
async def chat_completions(request: Request, token: str = Depends(verify_token)):
    """处理聊天请求"""
    body = await request.json()
    model_id = body.get("model")
    
    # 如果是 SageMaker 端点
    if "sagemaker" in model_id:
        return await invoke_sagemaker_endpoint(model_id, body)
    
    # 其他模型使用 Bedrock
    return await invoke_bedrock_model(model_id, body)


@app.post("/openai/verify")
async def verify(token: str = Depends(verify_token)):
    """OpenWebUI 验证端点"""
    return JSONResponse({
        "data": {
            "status": "success"
        },
        "object": "list"
    })


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=400)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录所有请求详情"""
    request_id = str(uuid.uuid4())
    logging.info(f"[{request_id}] Request: {request.method} {request.url}")
    
    response = await call_next(request)
    logging.info(f"[{request_id}] Response Status: {response.status_code}")
    
    return response

def clean_message(content: str, remove_think: bool = True) -> str:
    """清理消息中的特殊标记
    
    Args:
        content: 要清理的内容
        remove_think: 是否移除 think 标记中的内容
    """
    cleaned = content
    
    # 如果需要移除思考内容
    if remove_think:
        cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL)
    
    # 移除所有特殊标记
    markers = [
        "<\uff5cUser\uff5c>",
        "<\uff5cAssistant\uff5c>",
        "<\uff5cend\u2581of\u2581sentence\uff5c>",
        "<\uff5cbegin\u2581of\u2581sentence\uff5c>",
    ]
    for marker in markers:
        cleaned = cleaned.replace(marker, "")
    return cleaned.strip()

async def invoke_sagemaker_endpoint(model_id: str, body: dict):
    """调用 SageMaker 端点"""
    try:
        is_stream = body.get("stream", False)
        
        # 从模型 ID 中提取端点名称
        endpoint_name = model_id.split('/')[-1]
        
        # 获取消息
        messages = body.get("messages", [])
        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        # 构建完整的对话历史
        conversation = ""
        for i, msg in enumerate(messages[:-1]):  # 除了最后一条消息外的所有历史
            if not isinstance(msg, dict):
                raise HTTPException(status_code=400, detail="Invalid message format")
            role = msg.get("role", "")
            content = clean_message(msg.get("content", ""), remove_think=True)  # 移除历史消息中的思考内容
            if role == "user":
                conversation += "<\uff5cUser\uff5c>" + content + "\n"
            elif role == "assistant":
                conversation += "<\uff5cAssistant\uff5c>" + content + "<\uff5cend\u2581of\u2581sentence\uff5c>\n"
            
        # 处理最后一条消息
        last_msg = messages[-1]
        if not isinstance(last_msg, dict):
            raise HTTPException(status_code=400, detail="Invalid message format")
        
        if last_msg.get("role") == "user":
            conversation += "<\uff5cUser\uff5c>" + last_msg.get('content', '') + "\n<\uff5cAssistant\uff5c>"
        else:
            conversation += "<\uff5cAssistant\uff5c>" + clean_message(last_msg.get('content', ''), remove_think=True) + "<\uff5cend\u2581of\u2581sentence\uff5c>"
        
        # 准备输入数据
        input_data = {
            "inputs": conversation,
            "parameters": {
                "max_new_tokens": body.get("max_tokens", 4096),
                "temperature": body.get("temperature", 0.6),
                "top_p": body.get("top_p", 0.9),
                "top_k": None,
                "stop": [
                    "<\uff5cbegin\u2581of\u2581sentence\uff5c>",
                    "<\uff5cend\u2581of\u2581sentence\uff5c>",
                    "<\uff5cUser\uff5c>",
                    "<\uff5cAssistant\uff5c>"
                ]
            }
        }
        
        # 只记录最后一条消息的内容
        log_data = {
            "inputs": last_msg.get('content', ''),
            "parameters": input_data["parameters"]
        }
        logging.info(f"Calling SageMaker endpoint with data: {log_data}")
        
        # 调用 SageMaker 端点
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(input_data)
        )
        
        # 解析响应
        response_body = json.loads(response['Body'].read().decode())
        logging.info(f"SageMaker response: {response_body}")
        
        # 处理列表响应
        if isinstance(response_body, list) and response_body:
            generated_text = response_body[0].get('generated_text', '')
        else:
            generated_text = response_body.get('outputs', str(response_body))

        # 如果存在 <think>，只保留第一个 <think> 及之后的内容
        think_match = re.search(r'<think>.*$', generated_text, flags=re.DOTALL)
        if think_match:
            generated_text = think_match.group(0)

        # 清理模型响应，但保留思考内容
        generated_text = clean_message(generated_text, remove_think=False)

        if is_stream:
            async def stream_response():
                # 创建初始 chunk
                chunk = {
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": generated_text
                        },
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                
                # 发送结束 chunk
                chunk["choices"][0]["finish_reason"] = "stop"
                chunk["choices"][0]["delta"] = {}
                yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream"
            )
        else:
            return {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generated_text
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }
            
    except Exception as e:
        logging.error(f"Error invoking SageMaker endpoint: {e}")
        logging.error(f"Request body: {body}")
        raise HTTPException(status_code=500, detail=str(e))

async def invoke_bedrock_model(model_id: str, body: dict):
    """调用 Bedrock 模型"""
    try:
        # 准备输入数据
        input_data = {
            "prompt": body.get("messages", [])[-1].get("content", ""),
            "max_tokens": body.get("max_tokens", 256),
            "temperature": body.get("temperature", 1.0),
            "top_p": body.get("top_p", 0.9),
        }
        
        # 调用 Bedrock 模型
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            contentType='application/json',
            body=json.dumps(input_data)
        )
        
        # 解析响应
        response_body = json.loads(response['body'].read().decode())
        
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_body.get("completion", "")
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
    except Exception as e:
        logging.error(f"Error invoking Bedrock model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

handler = Mangum(app)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
