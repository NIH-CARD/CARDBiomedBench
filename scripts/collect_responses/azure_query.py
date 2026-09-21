import gc
import json
import os

from dotenv import load_dotenv
from openai import AzureOpenAI


class AzureQuery:
    """Query an Azure OpenAI chat-completions deployment."""

    def __init__(
        self,
        system_prompt,
        deployment_name,
        max_tokens,
        temperature,
        reasoning_effort=None,
    ):
        if not deployment_name:
            raise ValueError(
                "Azure OpenAI requires the configured model name to match a deployment name."
            )

        self.system_prompt = system_prompt
        self.model_name = deployment_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.cache_file = self.get_cache_file_path()
        self.cache = self.load_cache()
        self.client = self.initialize_azure_client()

    @staticmethod
    def initialize_azure_client():
        env_path = os.path.join(os.path.dirname(__file__), "../../configs/.env")
        load_dotenv(env_path)

        required_variables = {
            "AZURE_OPENAI_API_KEY": os.environ.get("AZURE_OPENAI_API_KEY"),
            "AZURE_OPENAI_ENDPOINT": os.environ.get("AZURE_OPENAI_ENDPOINT"),
            "AZURE_API_VERSION": os.environ.get("AZURE_API_VERSION"),
        }
        missing_variables = [
            name for name, value in required_variables.items() if not value
        ]
        if missing_variables:
            raise ValueError(
                "Missing Azure OpenAI environment variables: "
                + ", ".join(missing_variables)
            )

        return AzureOpenAI(
            api_key=required_variables["AZURE_OPENAI_API_KEY"],
            azure_endpoint=required_variables["AZURE_OPENAI_ENDPOINT"],
            api_version=required_variables["AZURE_API_VERSION"],
        )

    def get_cache_file_path(self):
        cache_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", ".cache", "model_responses_cache"
        )
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"azure_{self.model_name}_cache.json")

    def load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as cache_file:
                    return json.load(cache_file)
            except Exception as error:
                print(f"Error loading Azure response cache: {error}")
        return {}

    def save_cache(self):
        try:
            with open(self.cache_file, "w") as cache_file:
                json.dump(self.cache, cache_file)
        except Exception as error:
            print(f"Error saving Azure response cache: {error}")

    def get_cache_key(self, query: str):
        effort_suffix = (
            f"_effort={self.reasoning_effort}"
            if self.reasoning_effort is not None
            else ""
        )
        return f"azure_{self.model_name}{effort_suffix}_{self.system_prompt}_{query}"

    def query(self, query: str) -> str:
        cache_key = self.get_cache_key(query)
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            request_args = {
                "model": self.model_name,
                "max_completion_tokens": self.max_tokens,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": query},
                ],
            }
            if self.reasoning_effort is not None:
                request_args["reasoning_effort"] = self.reasoning_effort
            else:
                request_args["temperature"] = self.temperature

            completion = self.client.chat.completions.create(**request_args)
            response = completion.choices[0].message.content
            self.cache[cache_key] = response
            self.save_cache()
            return response
        except Exception as error:
            return f"Error in {self.model_name} response: {error}"

    def submit_batch_query(self, batch_file_path: str, metadata: dict = None) -> str:
        """Upload a JSONL request file and submit it as an Azure batch job."""
        try:
            with open(batch_file_path, "rb") as batch_file:
                batch_input_file = self.client.files.create(
                    file=batch_file,
                    purpose="batch",
                )

            batch = self.client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/chat/completions",
                completion_window="24h",
                metadata=metadata,
            )
            return batch.id
        except Exception as error:
            return {"error": f"Error during Azure batch submission: {error}"}

    def delete(self):
        try:
            if self.client is not None:
                self.client.close()
            del self.client
            gc.collect()
        except Exception as error:
            print(f"Error deleting Azure OpenAI client: {error}")
