import os
from dotenv import load_dotenv
from openai import OpenAI

class GPT4OQuery:
    def __init__(self, system_prompt):
        self.client = self.initialize_openai_client()
        self.system_prompt = system_prompt

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
            if openai_api_key:
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
                model='gpt-4o',
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": query}
                ]
            )
            response = chat_completion.choices[0].message.content
            return response
        except Exception as e:
            error_message = f"Error in GPT-4o response: {e}"
            return error_message