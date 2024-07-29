import os
import google.generativeai as genai
from . import SYSTEM_PROMPT

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def query_gemini(query: str) -> str:
    """
    Query the Google API with Gemini 1.5 Pro.

    Parameters:
    - query (str): The input query string.

    Returns:
    - str: The response content from the API or an error message.
    """
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-pro")
        chat = model.start_chat(history=[{"role": "user", "parts": [SYSTEM_PROMPT]}])
        response = chat.send_message(query)
        return response.text
    except Exception as e:
        error_message = f"Error in Gemini-1.5-Pro response: {e}"
        return error_message