from typing import Optional, Union, Any, Sequence

from llama_index import ServiceContext, VectorStoreIndex, Document
from llama_index.indices.base import BaseIndex
from llama_index.vector_stores.timescalevector import IndexType

from langflow import CustomComponent
from langflow.customer_base_class.document.llamaindexdocument import LLamaIndexDocument
from langflow.utils.logger import logger
from langflow.utils.converter import copy_object


class IndexingComponent(CustomComponent):
    display_name = "Indexing"
    description = "This Is The Component That Creates The DataIndex."

    def build_config(self):
        return {
            "llama_index_document": {
                "display_name": "LLamaIndexDocument",
                "required": True,
                "field_type": "LLamaIndexDocument",
            },
        }

    def build(
            self,
            llama_index_documents: list[LLamaIndexDocument],
    ) -> Union[IndexType, VectorStoreIndex, BaseIndex]:
        logger.info(f"llama_index_documents: {llama_index_documents}")
        documents = []
        for llama_index_document in llama_index_documents:
            print(llama_index_document)
            document = Document(
                id=llama_index_document.doc_id,
                text=llama_index_document.text,
                metadata=llama_index_document.metadata,
            )
            logger.info(f"llama_index_document11111: {llama_index_document}")
            documents.append(document)

        index = VectorStoreIndex.from_documents(documents)
        return index
