from typing import Optional, Union, Any

from llama_index import ServiceContext
from llama_index.llms.base import BaseLLM
from llama_index.llms.utils import LLMType

from langflow import CustomComponent
from langflow.customer_base_class.document.llamaindexdocument import LLamaIndexDocument


class ServiceContextComponent(CustomComponent):
    display_name = "ServiceContext"
    description = "The service context container is a utility container for LlamaIndex index and query classes.."

    def build_config(self):
        return {
            "llm": {
                "display_name": "LIOpenAI",
                "required": True,
                "field_type": "LIOpenAI",
            },
            "embed_model": {
                "display_name": "BaseEmbedding",
                "required": True,
                "field_type": "BaseEmbedding",
            },
        }

    def build(
            self,
            llm: Optional[LLMType] = "default",
            embed_model: Optional[Any] = "default",
    ) -> Union[ServiceContext]:
        service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model, )
        return service_context
