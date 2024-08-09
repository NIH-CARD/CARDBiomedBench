import os
import gc
from dotenv import load_dotenv
import anthropic

class ClaudeQuery:
    def __init__(self, system_prompt, model_name, max_tokens):
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.max_tokens = max_tokens
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
                return anthropic.Anthropic(api_key=anthropic_api_key)
            else:
                print("Anthropic API key not found in environment variables.")
        except Exception as e:
            print(f"Error initializing Claude model: {e}")
        return None

    def query(self, query: str) -> str:
        """
        Query the Claude API.

        Parameters:
        - query (str): The input query string.

        Returns:
        - str: The response content from the API or an error message.
        """
        try:
            message = self.model.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": query},
                ]
            )
            return message.content[0].text
        except Exception as e:
            error_message = f"Error in {self.model_name} response: {e}"
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