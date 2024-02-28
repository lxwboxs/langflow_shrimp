from typing import Optional

from langchain.chat_models.baidu_qianfan_endpoint import QianfanChatEndpoint
from langchain.llms.base import BaseLLM
from pydantic.v1 import SecretStr

from langflow import CustomComponent


class ProxyOpenAIChatEndpointComponent(CustomComponent):
    display_name: str = "ProxyOpenAIChatEndpoint"
    description: str = (
        "Proxy OpenAI chat models. Get more detail from "
        "https://python.langchain.com/docs/integrations/chat/baidu_qianfan_endpoint."
    )

    def build_config(self):
        return {
            "model_name": {
                "display_name": "Model Name",
                "options": [
                    "gpt-4-0125-preview",
                    "gpt-4-1106-preview",
                    "gpt-4",
                    "gpt-3.5-turbo-0125",
                    "gpt-3.5-turbo-1106",
                ],
                "info": "https://python.langchain.com/docs/integrations/chat/baidu_qianfan_endpoint",
                "required": True,
            },
            "openai_api_key": {
                "display_name": "OpenAI API Key",
                "required": True,
                "password": True,
                "info": "which you could get from  https://cloud.baidu.com/product/wenxinworkshop",
            },
            "temperature": {
                "display_name": "Temperature",
                "field_type": "float",
                "info": "Model params",
                "value": 0.95,
            },
            "openai_api_base": {
                "display_name": "Openai Api Base",
                "field_type": "str",
                "info": "Model params",
                "value": 1.0,
            },
            "openai_organization": {
                "display_name": "Openai Organization",
                "field_type": "str",
                "info": "adas",
            },
            "code": {"show": False},
        }

    def build(
        self,
        model_name: str = "gpt-3.5-turbo-0125",
        openai_api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        openai_api_base: Optional[float] = None,
        openai_organization: Optional[str] = None,
    ) -> BaseLLM:
        try:
            output = ChatOpenAI(  # type: ignore
                model_name =model_name,
                openai_api_key=openai_api_key,
                temperature=temperature,
                openai_api_base=openai_api_base,
                openai_organization=openai_organization,
            )
        except Exception as e:
            raise ValueError("Could not connect to Baidu Qianfan API.") from e
        return output  # type: ignore
