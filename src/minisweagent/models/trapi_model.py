import os
import logging
from dataclasses import dataclass
from typing import Any

from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
)
from openai import AzureOpenAI
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.litellm_model import LitellmModel

logger = logging.getLogger("trapi_model")


@dataclass
class TrapiModelConfig:
    model_name: str
    instance: str
    api_version: str
    scope: str
    trapi_url: str
    model_kwargs: dict[str, Any] = None

    def __post_init__(self):
        if self.model_kwargs is None:
            self.model_kwargs = {}

MODEL_DICT = {
    "gpt-4o": "gpt-4o_2024-11-20",
    "gpt-4_turbo": "gpt-4_turbo-2024-04-09",
    "gpt-4.1": "gpt-4.1_2025-04-14",
    "model-router": "model-router_2025-05-19",
    "o3": "o3_2025-04-16",
    "gpt-4o-mini": "gpt-4o-mini_2024-07-18",
    "o1": "o1_2024-12-17",
    "gpt-4.1-nano": "gpt-4.1-nano_2025-04-14",
    "gpt-4.1-mini": "gpt-4.1-mini_2025-04-14",
    "o4-mini": "o4-mini_2025-04-16",
    "text-embedding-3-large": "text-embedding-3-large_1",
    "text-embedding-3-small": "text-embedding-3-small_1",
    "computer-use-preview": "computer-use-preview_2025-03-11",
    "gpt-5": "gpt-5_2025-08-07",
    "gpt-5-mini": "gpt-5-mini_2025-08-07",
    "gpt-5-nano": "gpt-5-nano_2025-08-07",
    "gpt-5-chat": "gpt-5-chat_2025-08-07",
    "o1-mini": "o1-mini_2024-09-12",
    "gpt-image-1": "gpt-image-1",
    "grok-3": "grok-3_1",
}

class TrapiModel(LitellmModel):
    """Microsoft TRAPI (Research API) model that uses Azure authentication."""

    def __init__(self, **kwargs):
        # Extract TRAPI-specific config
        kwargs = {
            "model_name": kwargs['model_name'],
            "instance": kwargs.get("instance", os.environ['TRAPI_INSTANCE']),
            "api_version": kwargs.get("api_version", os.environ['TRAPI_API_VERSION']),
            "scope": kwargs.get("scope", os.environ['TRAPI_SCOPE']),
            "trapi_url": kwargs.get("trapi_url", os.environ['TRAPI_URL']),
            "model_kwargs": kwargs.get("model_kwargs", {}),
        }

        self.trapi_config = TrapiModelConfig(**kwargs)

        # Initialize parent with minimal config since we override _query
        super().__init__(model_name=self.trapi_config.model_name)
        
        # Set up TRAPI authentication and client
        scope = self.trapi_config.scope
        credential = get_bearer_token_provider(
            ChainedTokenCredential(
                AzureCliCredential(),
                ManagedIdentityCredential(),
            ),
            scope
        )
        
        endpoint = f"{self.trapi_config.trapi_url}/{self.trapi_config.instance}"
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=credential,
            api_version=self.trapi_config.api_version,
        )

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type((KeyboardInterrupt,)),
    )
    def _query(self, messages: list[dict[str, str]], **kwargs):
        """Override parent's _query to use TRAPI client instead of litellm."""
        merged_kwargs = self.trapi_config.model_kwargs | kwargs
        return self.client.chat.completions.create(
            model=MODEL_DICT[self.trapi_config.model_name],
            messages=messages,
            **merged_kwargs
        )

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        """Override to handle TRAPI response format and cost calculation."""
        response = self._query(messages, **kwargs)
        
        # For TRAPI, we'll estimate cost or set to 0 since litellm cost calculator may not work
        cost = 0.0  # TRAPI may have different cost structure
        
        self.n_calls += 1
        self.cost += cost
        GLOBAL_MODEL_STATS.add(cost)
        
        return {
            "content": response.choices[0].message.content or "",
        }
