from typing import Callable, Optional, Union

from langchain.chains import LLMChain

from langflow import CustomComponent
from langflow.customer_base_class.chain.my_custom_chain import MyCustomChain
from langflow.field_typing import (
    BaseLanguageModel,
    BaseMemory,
    BasePromptTemplate,
    Chain,
)


class MyCustomerChainComponent(CustomComponent):
    display_name = "MyCustomerChain"
    description = "Chain to run queries against MyCustomer"

    def build_config(self):
        return {
            "prompt": {"display_name": "Prompt"},
            "llm": {"display_name": "LLM"},
            "code": {"show": False},
        }

    def build(
        self,
        prompt: BasePromptTemplate,
        llm: BaseLanguageModel,
    ) -> Union[Chain, Callable, MyCustomChain]:
        return MyCustomChain(prompt=prompt, llm=llm,)
