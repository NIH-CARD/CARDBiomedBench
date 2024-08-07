import os
import gc
import time
from dotenv import load_dotenv
import google.generativeai as genai

class GeminiQuery:
    def __init__(self, system_prompt, model_name):
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.model = self.initialize_gemini_model()

    @staticmethod
    def initialize_gemini_model(self):
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
                return genai.GenerativeModel(model_name=self.model_name)
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
        time.sleep(10)
        try:
            chat = self.model.start_chat(
                history=[{"role": "user", "parts": [self.system_prompt]}]
            )
            response = chat.send_message(query)
            return response.text
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