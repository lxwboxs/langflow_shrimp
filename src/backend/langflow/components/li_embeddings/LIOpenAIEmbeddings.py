from typing import Any, Callable, Dict, List, Optional, Union

from llama_index import OpenAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbeddingMode, OpenAIEmbeddingModelType
from llama_index.llms.base import BaseLLM

from langflow import CustomComponent


class OpenAIEmbeddingsComponent(CustomComponent):
    display_name = "LIOpenAIEmbeddings"
    description = "OpenAI embedding models"

    def build_config(self):
        return {
            "embed_batch_size": {
                "display_name": "Embed Batch Size",
                "field_type": "int",
                "info": "Processing Batch Embedded Size (Default: 100)",
                "advanced": True,
            },
            "dimensions": {
                "display_name": "Dimensions",
                "field_type": "int",
                "info": "EmbeddedDimension (Default: 1536)",
                "advanced": True,
            },
            "mode": {
                "display_name": "Mode",
                "advanced": False,
                "options": ["similarity", "text_search", ],
            },
            "model": {
                "display_name": "Model",
                "advanced": False,
                "options": ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
            },
            "api_base": {"display_name": "OpenAI API Base", "password": True, "advanced": True},
            "api_key": {"display_name": "OpenAI API Key", "password": True},
        }

    def build(
            self,
            mode: str = OpenAIEmbeddingMode.TEXT_SEARCH_MODE,
            model: str = OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002,
            embed_batch_size: int = 100,
            dimensions: Optional[int] = None,
            api_key: Optional[str] = None,
            api_base: Optional[str] = None,
    ) -> Union[OpenAIEmbedding]:
        print(f"mode111111111: {mode}")
        return OpenAIEmbedding(
            mode=mode,
            model=model,
            embed_batch_size=embed_batch_size,
            dimensions=dimensions,
            api_key=api_key,
            api_base=api_base,
            max_retries=10,
            timeout=60.0,
            reuse_client=True,
        )
