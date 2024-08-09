import os
import gc
from dotenv import load_dotenv
from openai import OpenAI

class GPTQuery:
    def __init__(self, system_prompt, model_name, max_tokens):
        self.client = self.initialize_openai_client()
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.max_tokens = max_tokens

    @staticmethod
    def initialize_openai_client():
        """
        Initialize the OpenAI client.

        Returns:
        - OpenAI: Initialized OpenAI client.
        """
        try:
            load_dotenv(os.path.join(os.path.dirname(__file__), '../../configs/.env'))
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if (openai_api_key):
                return OpenAI(api_key=openai_api_key)
            else:
                print("OpenAI API key not found in environment variables.")
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
        return None

    def query(self, query: str) -> str:
        """
        Query the OpenAI API with GPT-4o.

        Parameters:
        - query (str): The input query string.

        Returns:
        - str: The response content from the API or an error message.
        """
        try:
            chat_completion = self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": query}
                ]
            )
            response = chat_completion.choices[0].message.content
            return response
        except Exception as e:
            error_message = f"Error in {self.model_name} response: {e}"
            return error_message
    
    def delete(self):
        """
        Delete the client to free up memory.
        """
        try:
            if self.client is not None:
                del self.client

            for attr in ['system_prompt', 'model_name']:
                if hasattr(self, attr):
                    delattr(self, attr)

            gc.collect()
        except Exception as e:
            print(f"Error during deletion of {self.model_name} client: {e}")