import os
import gc
import time
import requests
from dotenv import load_dotenv

class PerplexityQuery:
    def __init__(self, system_prompt, model_name, max_tokens):
        self.api_url = "https://api.perplexity.ai/chat/completions"
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.headers = self.initialize_headers()

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

    def query(self, query: str) -> str:
        """
        Query the Perplexity API.

        Parameters:
        - query (str): The input query string.

        Returns:
        - str: The response content from the API or an error message.
        """
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
            return response.json().get('choices', [{}])[0].get('message', {}).get('content', 'No content returned')
        except requests.exceptions.RequestException as e:
            return f"Error in {self.model_name} response: {e}"

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