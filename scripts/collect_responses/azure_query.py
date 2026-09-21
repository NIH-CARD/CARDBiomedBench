import gc
import json
import os

from dotenv import load_dotenv
from openai import AzureOpenAI


class AzureQuery:
    """Query an Azure OpenAI chat-completions deployment."""

    def __init__(self, system_prompt, deployment_name, max_tokens, temperature):
        if not deployment_name:
            raise ValueError(
                "Azure OpenAI requires the configured model name to match a deployment name."
            )

        self.system_prompt = system_prompt
        self.model_name = deployment_name
        self.max_tokens = max_tokens
        self.temperature = temperature
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
        return f"azure_{self.model_name}_{self.system_prompt}_{query}"

    def query(self, query: str) -> str:
        cache_key = self.get_cache_key(query)
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                max_completion_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": query},
                ],
            )
            response = completion.choices[0].message.content
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
            print(f"Error deleting Azure OpenAI client: {error}")
