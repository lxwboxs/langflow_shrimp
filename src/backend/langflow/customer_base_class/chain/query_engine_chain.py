from typing import Any, Dict, List, Optional

from langchain.chains.base import Chain
from langchain_core.callbacks import CallbackManagerForChainRun, AsyncCallbackManagerForChainRun
from llama_index.core.base_query_engine import BaseQueryEngine
from pydantic import Extra


class QueryEngineChain(Chain):
    """
    An example of a custom chain.
    """
    output_key: str = "text"  #: :meta private:
    query_engine: BaseQueryEngine

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    # 来自Chain抽象类，必须重写
    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return ["query"]


    # 来自Chain抽象类，必须重写
    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key]

    # 来自Chain抽象类，必须重写
    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        print("async _call")
        print(inputs)
        # Your custom chain logic goes here
        # This is just an example that mimics LLMChain
        response = self.query_engine.query(inputs["query"])
        # prompt_value = self.prompt.format_prompt(**inputs)

        # Whenever you call a language model, or another chain, you should pass
        # a callback manager to it. This allows the inner run to be tracked by
        # any callbacks that are registered on the outer run.
        # You can always obtain a callback manager for this by calling
        # `run_manager.get_child()` as shown below.
        # response = self.llm.generate_prompt(
        #     [prompt_value], callbacks=run_manager.get_child() if run_manager else None
        # )

        print(response)
        print(response.response)

        # If you want to log something about this run, you can do so by calling
        # methods on the `run_manager`, as shown below. This will trigger any
        # callbacks that are registered for that event.
        if run_manager:
            run_manager.on_text("Log something about this run")

        return {self.output_key: response.response}

    async def _acall(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # Your custom chain logic goes here
        # This is just an example that mimics LLMChain
        prompt_value = self.prompt.format_prompt(**inputs)

        # Whenever you call a language model, or another chain, you should pass
        # a callback manager to it. This allows the inner run to be tracked by
        # any callbacks that are registered on the outer run.
        # You can always obtain a callback manager for this by calling
        # `run_manager.get_child()` as shown below.
        response = await self.query_engine.query(inputs["query"])

        # If you want to log something about this run, you can do so by calling
        # methods on the `run_manager`, as shown below. This will trigger any
        # callbacks that are registered for that event.
        if run_manager:
            await run_manager.on_text("Log something about this run")

        return {self.output_key: response.response}

    @property
    def _chain_type(self) -> str:
        return "query_engine_chain"