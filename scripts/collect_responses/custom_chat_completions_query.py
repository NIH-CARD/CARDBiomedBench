import gc
import json
import os
from hashlib import sha256
from urllib.parse import urlsplit

from dotenv import load_dotenv
from openai import OpenAI

from scripts.collect_responses.cache_utils import (
    filter_valid_cache,
    is_valid_cached_response,
)


class CustomChatCompletionsQuery:
    """Query an arbitrary OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        system_prompt,
        model_name,
        endpoint_env=None,
        token_env=None,
        extra_body_env=None,
    ):
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.endpoint = self._load_setting("endpoint", endpoint_env)
        token = self._load_setting("bearer token", token_env)
        self.extra_body = self._load_extra_body(extra_body_env)
        self.cache_file = self.get_cache_file_path()
        self.cache = self.load_cache()
        self.client = OpenAI(api_key=token, base_url=self.endpoint)

    @staticmethod
    def _load_setting(label, env_name):
        load_dotenv(
            os.path.join(os.path.dirname(__file__), "../../configs/.env")
        )
        resolved = os.environ.get(env_name) if env_name else None
        if not resolved:
            raise ValueError(
                f"Missing custom chat-completions {label} environment variable "
                f"{env_name!r}"
            )
        return resolved

    @classmethod
    def _load_extra_body(cls, env_name):
        value = cls._load_setting("extra body", env_name)
        try:
            extra_body = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Custom chat-completions extra body environment variable "
                f"{env_name!r} must contain valid JSON"
            ) from error
        if not isinstance(extra_body, dict):
            raise ValueError(
                f"Custom chat-completions extra body environment variable "
                f"{env_name!r} must contain a JSON object"
            )
        return extra_body

    def get_cache_file_path(self):
        cache_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", ".cache", "model_responses_cache"
        )
        os.makedirs(cache_dir, exist_ok=True)
        endpoint_id = sha256(self.endpoint.encode()).hexdigest()[:12]
        safe_model = self.model_name.replace("/", "__")
        return os.path.join(
            cache_dir, f"custom_{safe_model}_{endpoint_id}_cache.json"
        )

    def load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as cache_file:
                    return filter_valid_cache(json.load(cache_file))
            except Exception as error:
                print(f"Error loading custom endpoint response cache: {error}")
        return {}

    def save_cache(self):
        try:
            with open(self.cache_file, "w") as cache_file:
                json.dump(self.cache, cache_file)
        except Exception as error:
            print(f"Error saving custom endpoint response cache: {error}")

    def get_cache_key(self, query):
        request_config = json.dumps(self.extra_body, sort_keys=True)
        return f"custom_{self.model_name}_{request_config}_{self.system_prompt}_{query}"

    def query(self, query):
        cache_key = self.get_cache_key(query)
        cached_response = self.cache.get(cache_key)
        if is_valid_cached_response(cached_response):
            return cached_response
        self.cache.pop(cache_key, None)

        try:
            request_args = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": query},
                ],
            }
            if self.extra_body:
                request_args["extra_body"] = self.extra_body

            completion = self.client.chat.completions.create(**request_args)
            choice = completion.choices[0]
            response = choice.message.content
            if not is_valid_cached_response(response):
                return (
                    f"Error in {self.model_name} response: empty message content"
                    f"; finish_reason={getattr(choice, 'finish_reason', None)!r}"
                )

            self.cache[cache_key] = response
            self.save_cache()
            return response
        except Exception as error:
            endpoint_name = urlsplit(self.endpoint).netloc or self.endpoint
            return f"Error in {self.model_name} response from {endpoint_name}: {error}"

    def delete(self):
        try:
            if self.client is not None:
                self.client.close()
            del self.client
            gc.collect()
        except Exception as error:
            print(f"Error deleting custom endpoint client: {error}")
