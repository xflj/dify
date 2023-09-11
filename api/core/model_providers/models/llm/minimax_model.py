from _decimal import Decimal
import re, string

from typing import List, Optional, Any

from langchain.callbacks.manager import Callbacks
from langchain.llms import Minimax
from langchain.schema import LLMResult

from core.model_providers.error import LLMBadRequestError
from core.model_providers.models.llm.base import BaseLLM
from core.model_providers.models.entity.message import PromptMessage, MessageType
from core.model_providers.models.entity.model_params import ModelMode, ModelKwargs


class MinimaxModel(BaseLLM):
    model_mode: ModelMode = ModelMode.COMPLETION

    def _init_client(self) -> Any:
        provider_model_kwargs = self._to_model_kwargs_input(self.model_rules, self.model_kwargs)
        return Minimax(
            model=self.name,
            model_kwargs={
                'stream': False
            },
            callbacks=self.callbacks,
            **self.credentials,
            **provider_model_kwargs
        )

    def _run(self, messages: List[PromptMessage],
             stop: Optional[List[str]] = None,
             callbacks: Callbacks = None,
             **kwargs) -> LLMResult:
        """
        run predict by prompt messages and stop words.

        :param messages:
        :param stop:
        :param callbacks:
        :return:
        """
        prompts = self._get_prompt_from_messages(messages)
        return self._client.generate([prompts], stop, callbacks)

    def get_num_tokens(self, messages: List[PromptMessage]) -> int:
        """
        get num tokens of prompt messages.

        :param messages:
        :return:
        """
        prompts = self._get_prompt_from_messages(messages)
        return max(self._get_num_tokens(prompts), 0)

    def get_currency(self):
        return 'RMB'

    def _set_model_kwargs(self, model_kwargs: ModelKwargs):
        provider_model_kwargs = self._to_model_kwargs_input(self.model_rules, model_kwargs)
        for k, v in provider_model_kwargs.items():
            if hasattr(self.client, k):
                setattr(self.client, k, v)

    def handle_exceptions(self, ex: Exception) -> Exception:
        if isinstance(ex, ValueError):
            return LLMBadRequestError(f"Minimax: {str(ex)}")
        else:
            return ex

    def _get_num_tokens(self, text: str) -> float:
        """Calculate number of tokens."""
        total = Decimal(0)
        words = re.findall(r'\b\w+\b|[{}]|\s'.format(re.escape(string.punctuation)), text)
        for word in words:
            if word:
                if '\u4e00' <= word <= '\u9fff':  # if chinese
                    total += Decimal('1.5')
                else:
                    total += Decimal('0.8')
        return int(total)
