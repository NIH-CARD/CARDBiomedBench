import os
import gc
from dotenv import load_dotenv
import anthropic

class ClaudeQuery:
    def __init__(self, system_prompt, model_name):
        self.system_prompt = system_prompt
        self.model_name = model_name
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
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": self.system_prompt},
                    {"role": "user", "content": query}
                ]
            )
            return message.content
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
            print(f"{self.model_name} model and attributes deleted successfully.")
        except Exception as e:
            print(f"Error during deletion of {self.model_name} model: {e}")