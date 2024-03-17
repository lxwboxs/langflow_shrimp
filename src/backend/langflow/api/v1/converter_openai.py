import time
from typing import Annotated, List, Optional, Union

from fastapi import APIRouter, Body, Depends
from pydantic import BaseModel

from langflow.api.v1.endpoints import process
from langflow.api.v1.schemas import (
    ProcessResponse,
)
from langflow.services.auth.utils import api_key_security
from langflow.services.database.models.user.model import User
from langflow.services.deps import get_session, get_session_service, get_task_service
from langflow.services.session.service import SessionService

try:
    from langflow.worker import process_graph_cached_task
except ImportError:

    def process_graph_cached_task(*args, **kwargs):
        raise NotImplementedError("Celery is not installed")

from langflow.services.task.service import TaskService
from sqlmodel import Session

# build router
router = APIRouter(tags=["Converter"])

# OpenAI 接口参数的默认值
openai_default_params = {
    "model": "text-davinci-002",
    "prompt": "",
    "temperature": 1.0,
    "max_tokens": 100
}

# OpenAI 接口响应的示例结构
openai_response_template = {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "gpt-3.5-turbo-0125",
    "system_fingerprint": "fp_44709d6fcb",
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "\n\nHello there, how may I assist you today?",
        },
        "logprobs": None,
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 9,
        "completion_tokens": 12,
        "total_tokens": 21
    }
}


# OpenAI 风格的响应体模型
class OpenAIResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[dict]
    usage: dict
    # result: str  # 添加 result 字段


# @router.post(
#     "/converter/process/{flow_id}",
#     response_model=ProcessResponse,
# )
# async def process(
#         # ... 其他参数保持不变
# ):
#     # ... 省略了之前的代码逻辑
#
#     # 将原始输入转换为 OpenAI 的参数格式
#     openai_params = {**openai_default_params, **inputs}
#
#     # 这里是调用 OpenAI 接口的伪代码
#     # openai_response = openai.Completion.create(**openai_params)
#
#     # 假设 openai_response 是 OpenAI 接口的响应
#     # 我们需要将其转换为我们的 ProcessResponse 格式
#     # 以下是一个简化的转换示例
#     openai_response = openai_response_template  # 假设这是 OpenAI 的响应
#     process_response = ProcessResponse(
#         result=openai_response["choices"][0]["text"],  # 假设我们只关心第一个选择的文本
#         status="completed",  # 假设任务已经完成
#         task=None,  # OpenAI 响应中没有任务信息
#         session_id=session_id,  # 保持原有的 session_id
#         backend="openai"  # 指定后端为 OpenAI
#     )
#
#     # ... 省略了之后的代码逻辑
#
#     return process_response


@router.post(
    "/converter/process_openai/{flow_id}",
    response_model=OpenAIResponse,
)
async def process_openai(
        session: Annotated[Session, Depends(get_session)],
        flow_id: str,
        inputs: Optional[Union[List[dict], dict]] = None,
        tweaks: Optional[dict] = None,
        clear_cache: Annotated[bool, Body(embed=True)] = False,  # noqa: F821
        session_id: Annotated[Union[None, str], Body(embed=True)] = None,  # noqa: F821
        task_service: "TaskService" = Depends(get_task_service),
        api_key_user: User = Depends(api_key_security),
        sync: Annotated[bool, Body(embed=True)] = True,  # noqa: F821
        session_service: SessionService = Depends(get_session_service),
):
    process_response = await process(session, flow_id, inputs, tweaks, clear_cache, session_id, task_service,
                                     api_key_user, sync,
                                     session_service)

    # 将原始输出转换为openai的格式

    # 创建 OpenAI 风格的响应体
    openai_response = OpenAIResponse(
        id=str(id(process_response)),
        object="text_completion",
        created=int(time.time()),  # 假设现在的时间作为创建时间
        model="your-model-name",  # 你需要指定使用的模型名称
        choices=[
            {
                "message": {
                    "role": "assistant",
                    "content": process_response.result['text']
                },
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop"
            }
        ],
        usage={
            "prompt_tokens": 0,  # 假设没有提示 tokens
            "completion_tokens": len(process_response.result['text'].split()),  # 计算完成 tokens
            "total_tokens": len(process_response.result['text'].split()),  # 总 tokens 就是完成 tokens
        },
        # result='Transformed result content'  # 添加 result 字段的值
    )
    return openai_response
