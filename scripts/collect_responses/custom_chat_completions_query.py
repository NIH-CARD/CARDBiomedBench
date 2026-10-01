import gc
import json
import os
from hashlib import sha256
from urllib.parse import urlsplit

from dotenv import load_dotenv
from openai import OpenAI

from scripts.collect_responses.cache_utils import (
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
        trace_file_path=None,
    ):
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.endpoint = self._load_setting("endpoint", endpoint_env)
        token = self._load_setting("bearer token", token_env)
        self.extra_body = self._load_extra_body(extra_body_env)
        self.trace_file_path = trace_file_path
        self.trace_ids = self._load_trace_ids()
        self.last_trace_id = None
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
                    cache = json.load(cache_file)
                if not isinstance(cache, dict):
                    raise ValueError("Response cache is not a JSON object")
                return {
                    key: value
                    for key, value in cache.items()
                    if self._get_cached_content(value) is not None
                }
            except Exception as error:
                print(f"Error loading custom endpoint response cache: {error}")
        return {}

    @staticmethod
    def _get_cached_content(entry):
        if is_valid_cached_response(entry):
            return entry
        if isinstance(entry, dict) and is_valid_cached_response(entry.get("content")):
            return entry["content"]
        return None

    def _load_trace_ids(self):
        trace_ids = set()
        if not self.trace_file_path or not os.path.exists(self.trace_file_path):
            return trace_ids
        try:
            with open(self.trace_file_path, "r") as trace_file:
                for line_number, line in enumerate(trace_file, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        print(
                            f"Skipping malformed trace record at "
                            f"{self.trace_file_path}:{line_number}: {error}"
                        )
                        continue
                    if isinstance(record, dict) and record.get("id"):
                        trace_ids.add(record["id"])
        except Exception as error:
            raise ValueError(
                f"Could not read trace file {self.trace_file_path}: {error}"
            ) from error
        return trace_ids

    @staticmethod
    def _json_value(value):
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        return value

    def _save_trace(self, completion):
        completion_data = self._json_value(completion)
        trace = (
            completion_data.get("trace")
            if isinstance(completion_data, dict)
            else getattr(completion, "trace", None)
        )
        completion_id = getattr(completion, "id", None)
        if not trace or not completion_id or not self.trace_file_path:
            return None
        if completion_id in self.trace_ids:
            return completion_id

        record = {
            "id": completion_id,
            "created": getattr(completion, "created", None),
            "model": getattr(completion, "model", None),
            "usage": (
                completion_data.get("usage")
                if isinstance(completion_data, dict)
                else self._json_value(getattr(completion, "usage", None))
            ),
            "trace": trace,
        }
        try:
            os.makedirs(os.path.dirname(self.trace_file_path), exist_ok=True)
            with open(self.trace_file_path, "a") as trace_file:
                trace_file.write(json.dumps(record) + "\n")
                trace_file.flush()
            self.trace_ids.add(completion_id)
            return completion_id
        except Exception as error:
            print(f"Error saving custom endpoint trace: {error}")
            return None

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
        self.last_trace_id = None
        cache_key = self.get_cache_key(query)
        cached_entry = self.cache.get(cache_key)
        cached_response = self._get_cached_content(cached_entry)
        if cached_response is not None:
            if isinstance(cached_entry, dict):
                cached_trace_id = cached_entry.get("trace_id")
                if cached_trace_id in self.trace_ids:
                    self.last_trace_id = cached_trace_id
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

            self.last_trace_id = self._save_trace(completion)
            self.cache[cache_key] = {
                "content": response,
                "trace_id": self.last_trace_id,
            }
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
