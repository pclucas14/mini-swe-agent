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
            model=self.trapi_config.model_name,
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
