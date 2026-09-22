import gc
import json
import os

from dotenv import load_dotenv
from openai import OpenAI


class OpenRouterQuery:
    """Query an OpenRouter model through its OpenAI-compatible API."""

    VALID_REASONING_EFFORTS = {"minimal", "low", "medium", "high", "xhigh", "max"}

    def __init__(self, system_prompt, model_name, max_tokens, temperature,
                 reasoning_effort=None):
        if (
            reasoning_effort is not None
            and reasoning_effort not in self.VALID_REASONING_EFFORTS
        ):
            raise ValueError(
                f"Invalid OpenRouter reasoning effort '{reasoning_effort}'."
            )

        self.system_prompt = system_prompt
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.cache_file = self.get_cache_file_path()
        self.cache = self.load_cache()
        self.client = self.initialize_openrouter_client()

    @staticmethod
    def initialize_openrouter_client():
        env_path = os.path.join(os.path.dirname(__file__), "../../configs/.env")
        load_dotenv(env_path)

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing OpenRouter environment variable: OPENROUTER_API_KEY"
            )

        return OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    def get_cache_file_path(self):
        cache_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", ".cache", "model_responses_cache"
        )
        os.makedirs(cache_dir, exist_ok=True)
        cache_name = self.model_name.replace("/", "__")
        return os.path.join(cache_dir, f"openrouter_{cache_name}_cache.json")

    def load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as cache_file:
                    cache = json.load(cache_file)
                if not isinstance(cache, dict):
                    raise ValueError("OpenRouter response cache is not a JSON object")

                # Older runs could cache a successful API response whose
                # message.content was null. Do not reuse those entries; they
                # should be queried again.
                return {
                    key: value
                    for key, value in cache.items()
                    if isinstance(value, str) and value.strip()
                }
            except Exception as error:
                print(f"Error loading OpenRouter response cache: {error}")
        return {}

    def save_cache(self):
        try:
            with open(self.cache_file, "w") as cache_file:
                json.dump(self.cache, cache_file)
        except Exception as error:
            print(f"Error saving OpenRouter response cache: {error}")

    def get_cache_key(self, query: str):
        effort_suffix = (
            f"_effort={self.reasoning_effort}"
            if self.reasoning_effort is not None
            else ""
        )
        return (
            f"openrouter_{self.model_name}{effort_suffix}_"
            f"{self.system_prompt}_{query}"
        )

    def query(self, query: str) -> str:
        cache_key = self.get_cache_key(query)
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            request_args = dict(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": query},
                ],
            )
            if self.reasoning_effort is not None:
                request_args["reasoning_effort"] = self.reasoning_effort

            completion = self.client.chat.completions.create(**request_args)
            choice = completion.choices[0]
            response = choice.message.content
            if not isinstance(response, str) or not response.strip():
                usage = getattr(completion, "usage", None)
                completion_details = getattr(
                    usage, "completion_tokens_details", None
                )
                diagnostic = (
                    "empty message content"
                    f"; finish_reason={getattr(choice, 'finish_reason', None)!r}"
                    f"; completion_tokens={getattr(usage, 'completion_tokens', None)!r}"
                    f"; reasoning_tokens={getattr(completion_details, 'reasoning_tokens', None)!r}"
                )
                return f"Error in {self.model_name} response: {diagnostic}"

            self.cache[cache_key] = response
            self.save_cache()
            return response
        except Exception as error:
            return f"Error in {self.model_name} response: {error}"

    def delete(self):
        try:
            if self.client is not None:
                self.client.close()
            del self.client
            gc.collect()
        except Exception as error:
            print(f"Error deleting OpenRouter client: {error}")
