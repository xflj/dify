from _decimal import Decimal
import re, string

from langchain.embeddings import MiniMaxEmbeddings

from core.model_providers.error import LLMBadRequestError
from core.model_providers.models.embedding.base import BaseEmbedding
from core.model_providers.providers.base import BaseModelProvider


class MinimaxEmbedding(BaseEmbedding):
    def __init__(self, model_provider: BaseModelProvider, name: str):
        credentials = model_provider.get_model_credentials(
            model_name=name,
            model_type=self.type
        )

        client = MiniMaxEmbeddings(
            model=name,
            **credentials
        )

        super().__init__(model_provider, client, name)

    def handle_exceptions(self, ex: Exception) -> Exception:
        if isinstance(ex, ValueError):
            return LLMBadRequestError(f"Minimax: {str(ex)}")
        else:
            return ex

    def get_num_tokens(self, text: str) -> float:
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
