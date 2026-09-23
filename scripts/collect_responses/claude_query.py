import os
import gc
import json
from dotenv import load_dotenv
import anthropic

from scripts.collect_responses.cache_utils import (
    filter_valid_cache,
    is_valid_cached_response,
)

class ClaudeQuery:
    VALID_EFFORTS = {"low", "medium", "high", "xhigh", "max"}

    def __init__(self, system_prompt, model_name, max_tokens, temperature,
                 thinking_budget_tokens=None, effort=None):
        if thinking_budget_tokens is not None and effort is not None:
            raise ValueError("Use either thinking_budget_tokens or effort, not both.")
        if effort is not None and effort not in self.VALID_EFFORTS:
            raise ValueError(
                f"Invalid Claude effort '{effort}'. Expected one of: "
                f"{', '.join(sorted(self.VALID_EFFORTS))}."
            )

        self.system_prompt = system_prompt
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.thinking_budget_tokens = thinking_budget_tokens
        self.effort = effort
        self.cache_file = self.get_cache_file_path()
        self.cache = self.load_cache()
        self.model = self.initialize_claude_model()

    @staticmethod
    def initialize_claude_model():
        """
        Initialize an anthropic model.

        Returns:
        - anthropic.Anthropic: Initialized anthropic model.
        """
        try:
            load_dotenv(os.path.join(os.path.dirname(__file__), '../../configs/.env'))
            anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
            if anthropic_api_key:
                return anthropic.Anthropic(api_key=anthropic_api_key, timeout=120)
            else:
                print("Anthropic API key not found in environment variables.")
        except Exception as e:
            print(f"Error initializing Claude model: {e}")
        return None

    def get_cache_file_path(self):
        """
        Get the path to the cache file based on the model name.

        Returns:
        - str: The cache file path.
        """
        cache_dir = os.path.join(os.path.dirname(__file__), '..', '..', '.cache', 'model_responses_cache')
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f'{self.model_name}_cache.json')

    def load_cache(self):
        """
        Load the cache from the cache file.
        
        Returns:
        - dict: The loaded cache data.
        """
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return filter_valid_cache(json.load(f))
            except Exception as e:
                print(f"Error loading cache file: {e}")
        return {}

    def save_cache(self):
        """
        Save the cache to a cache file.
        """
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            print(f"Error saving cache file: {e}")

    def get_cache_key(self, query: str):
        """
        Generate a unique cache key based on the model name, query, and system prompt.

        Parameters:
        - query (str): The input query string.

        Returns:
        - str: The cache key.
        """
        effort_suffix = f"_effort={self.effort}" if self.effort is not None else ""
        return f"{self.model_name}{effort_suffix}_{self.system_prompt}_{query}"

    def query(self, query: str) -> str:
        """
        Query the Claude API or retrieve from cache if available.

        Parameters:
        - query (str): The input query string.

        Returns:
        - str: The response content from the API or the cached response.
        """
        cache_key = self.get_cache_key(query)

        # Check if the result is already cached
        cached_response = self.cache.get(cache_key)
        if is_valid_cached_response(cached_response):
            return cached_response
        self.cache.pop(cache_key, None)

        # If not cached, query the API
        try:
            kwargs = dict(
                model=self.model_name,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                messages=[{"role": "user", "content": query}],
            )

            # Enable thinking automatically if budget set
            if self.thinking_budget_tokens is not None:
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": self.thinking_budget_tokens}
                kwargs["temperature"] = 1.0
            elif self.effort is not None:
                # Claude 5 models use adaptive thinking. It is enabled by
                # default, and effort controls its depth and token usage.
                kwargs["output_config"] = {"effort": self.effort}
            else:
                kwargs["temperature"] = self.temperature

            message = self.model.messages.create(**kwargs)
            text_parts = [
                getattr(block, "text", "")
                for block in message.content
                if getattr(block, "type", None) == "text"
            ]
            response = "\n".join([t for t in text_parts if t]).strip()
            if not is_valid_cached_response(response):
                return (
                    f"Error in {self.model_name} response: empty message content"
                    f"; stop_reason={getattr(message, 'stop_reason', None)!r}"
                )

            # Cache the result
            self.cache[cache_key] = response
            self.save_cache()

            return response
        except Exception as e:
            error_message = f"Error in {self.model_name} response: {e}"
            print(error_message)
            return error_message

    def delete(self):
        """
        Delete the model to free up memory.
        """
        try:
            if self.model is not None:
                del self.model

            for attr in ['system_prompt', 'model_name']:
                if hasattr(self, attr):
                    delattr(self, attr)

            gc.collect()
        except Exception as e:
            print(f"Error during deletion of {self.model_name} model: {e}")
