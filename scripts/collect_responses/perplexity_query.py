import os
import gc
import time
import json
import requests
from dotenv import load_dotenv

class PerplexityQuery:
    def __init__(self, system_prompt, model_name, max_tokens):
        self.api_url = "https://api.perplexity.ai/chat/completions"
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.headers = self.initialize_headers()
        self.cache_file = self.get_cache_file_path()
        self.cache = self.load_cache()

    @staticmethod
    def initialize_headers():
        """
        Initialize the headers for the Perplexity API, including the authorization token.

        Returns:
        - dict: Headers with content-type, accept, and authorization bearer token.
        """
        try:
            load_dotenv(os.path.join(os.path.dirname(__file__), '../../configs/.env'))
            perplexity_api_key = os.environ.get("PERPLEXITY_API_KEY")
            headers = {
                "accept": "application/json",
                "content-type": "application/json",
            }
            if perplexity_api_key:
                headers["authorization"] = f"Bearer {perplexity_api_key}"
            else:
                print("Perplexity API key not found in environment variables.")
            return headers
        except Exception as e:
            print(f"Error initializing headers: {e}")
            return {}

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
                    return json.load(f)
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
        return f"{self.model_name}_{self.system_prompt}_{query}"

    def query(self, query: str) -> str:
        """
        Query the Perplexity API or retrieve from cache if available.

        Parameters:
        - query (str): The input query string.

        Returns:
        - str: The response content from the API or the cached response.
        """
        cache_key = self.get_cache_key(query)

        # Check if the result is already cached
        if cache_key in self.cache:
            return self.cache[cache_key]

        # If not cached, query the API
        payload = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": query}
            ]
        }

        try:
            time.sleep(3)
            response = requests.post(self.api_url, json=payload, headers=self.headers)
            response.raise_for_status()
            response_content = response.json().get('choices', [{}])[0].get('message', {}).get('content', 'No content returned')

            # Cache the result
            self.cache[cache_key] = response_content
            self.save_cache()

            return response_content
        except requests.exceptions.RequestException as e:
            error_message = f"Error in {self.model_name} response: {e}"
            return error_message

    def delete(self):
        """
        Delete the attributes to free up memory.
        """
        try:
            for attr in ['system_prompt', 'model_name', 'headers']:
                if hasattr(self, attr):
                    delattr(self, attr)

            gc.collect()
        except Exception as e:
            print(f"Error during deletion of Perplexity client: {e}")