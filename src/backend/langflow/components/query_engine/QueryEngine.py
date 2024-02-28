from typing import Optional, Union, Any, Sequence

from llama_index import ServiceContext, VectorStoreIndex, Document
from llama_index.core.base_query_engine import BaseQueryEngine
from llama_index.indices.base import BaseIndex

from langflow import CustomComponent
from langflow.customer_base_class.document.llamaindexdocument import LLamaIndexDocument


class QueryEngineComponent(CustomComponent):
    display_name = "QueryEngine"
    description = "This Is A QueryEngine."

    def build_config(self):
        return {
            # "index": {
            #     "display_name": "BaseIndex",
            #     # "required": False,
            #     "field_type": "BaseIndex",
            # },
            "llama_index_document": {
                "display_name": "LLamaIndexDocument",
                "required": True,
                "field_type": "LLamaIndexDocument",
            },
            "service_context": {
                "display_name": "ServiceContext",
                "required": True,
                "field_type": "ServiceContext",
            },
            "similarity_top_k": {
                "display_name": "SimilarityTopK",
                "field_type": "int",
                "info": "How many blocks of data before the similarity is taken out.",
                "advanced": True,
                "show": True,
                "value": 5,
            },
            "streaming": {
                "display_name": "Streaming",
                "field_type": "bool",
                "info": "whether Streaming Return Is Required.",
                "advanced": True,
                "value": False,
            },
        }

    def build(
            self,
            # base_index: BaseIndex,
            llama_index_documents: list[LLamaIndexDocument],
            service_context: Optional[ServiceContext] = None,
            similarity_top_k: Optional[int] = 3,
            streaming: Optional[bool] = True,

    ) -> Union[BaseQueryEngine]:
        # if base_index is None:
        documents = []
        for llama_index_document in llama_index_documents:
            document = Document(
                id=llama_index_document.doc_id,
                text=llama_index_document.text,
                metadata=llama_index_document.metadata,
            )
            documents.append(document)
        base_index = VectorStoreIndex.from_documents(documents, service_context=service_context, show_progress=True)
        query_engine = base_index.as_query_engine(service_context=service_context, similarity_top_k=similarity_top_k,
                                                  streaming=streaming, )
        return query_engine
