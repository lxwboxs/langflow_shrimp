from typing import Callable, Optional, Union

from langchain.chains import LLMChain
from llama_index.core.base_query_engine import BaseQueryEngine

from langflow import CustomComponent
from langflow.customer_base_class.chain.my_custom_chain import MyCustomChain
from langflow.customer_base_class.chain.query_engine_chain import QueryEngineChain
from langflow.field_typing import (
    BaseLanguageModel,
    BaseMemory,
    BasePromptTemplate,
    Chain,
)


class QueryEngineChainComponent(CustomComponent):
    display_name = "QueryEngineChain"
    description = "Chain to run queries against MyCustomer"

    def build_config(self):
        return {
            "query_engine": {
                "display_name": "BaseQueryEngine",
                "required": True,
                "field_type": "BaseQueryEngine",
            },
            "code": {"show": False},
        }

    def build(
        self,
        query_engine: BaseQueryEngine,
    ) -> Union[Chain, Callable, QueryEngineChain]:
        return QueryEngineChain(query_engine=query_engine,)
