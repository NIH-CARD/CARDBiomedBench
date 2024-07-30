import os
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai
from scripts import MODELS

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

# Initialize clients/models
OPENAI_CLIENT = initialize_openai_client() if 'gpt-4o' in MODELS else None
GEMINI_MODEL = initialize_gemini_model() if 'gemini-1.5-pro' in MODELS else None