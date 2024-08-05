import os
import gc
from dotenv import load_dotenv
import google.generativeai as genai

class GeminiQuery:
    def __init__(self, system_prompt):
        self.model = self.initialize_gemini_model()
        self.system_prompt = system_prompt

    @staticmethod
    def initialize_gemini_model():
        """
        Initialize the Gemini model.

        Returns:
        - genai.GenerativeModel: Initialized Gemini model.
        """
        try:
            load_dotenv(os.path.join(os.path.dirname(__file__), '../../configs/.env'))
            google_api_key = os.environ["GOOGLE_API_KEY"]
            if google_api_key:
                genai.configure(api_key=google_api_key)
                return genai.GenerativeModel(model_name="gemini-1.5-pro")
            else:
                print("Google API key not found in environment variables.")
        except Exception as e:
            print(f"Error initializing Gemini model: {e}")
        return None

    def query(self, query: str) -> str:
        """
        Query the Google API with Gemini 1.5 Pro.

        Parameters:
        - query (str): The input query string.

        Returns:
        - str: The response content from the API or an error message.
        """
        try:
            chat = self.model.start_chat(
                history=[{"role": "user", "parts": [self.system_prompt]}]
            )
            response = chat.send_message(query)
            return response.text
        except Exception as e:
            error_message = f"Error in Gemini-1.5-Pro response: {e}"
            return error_message
    
    def delete(self):
        """
        Delete the model to free up memory.
        """
        try:
            if self.model is not None:
                del self.model

            # Explicitly delete other attributes if necessary
            for attr in ['system_prompt']:
                if hasattr(self, attr):
                    delattr(self, attr)

            # Clear any remaining references
            gc.collect()
            print(f"Gemini-1.5-Pro model and attributes deleted successfully.")
        except Exception as e:
            print(f"Error during deletion of Gemini-1.5-Pro model: {e}")
