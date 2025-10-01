import os
import gc
import time
import json
from dotenv import load_dotenv
from google import genai

class GeminiQuery:
    def __init__(self, system_prompt, model_name, max_tokens, temperature, thinking_budget=-1):
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.thinking_budget = thinking_budget
        self.cache_file = self.get_cache_file_path()
        self.cache = self.load_cache()
        self.model = self.initialize_gemini_model()

    def initialize_gemini_model(self):
        try:
            load_dotenv(os.path.join(os.path.dirname(__file__), '../../configs/.env'))
            google_api_key = os.environ.get("GOOGLE_API_KEY")
            if not google_api_key:
                print("Google API key not found in environment variables.")
                return None
            return genai.Client(api_key=google_api_key)
        except Exception as e:
            print(f"Error initializing Gemini client: {e}")
        return None

    def get_cache_file_path(self):
        cache_dir = os.path.join(os.path.dirname(__file__), '..', '..', '.cache', 'model_responses_cache')
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f'{self.model_name}_cache.json')

    def load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading cache file: {e}")
        return {}

    def save_cache(self):
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            print(f"Error saving cache file: {e}")

    def get_cache_key(self, query: str):
        return f"{self.model_name}_{self.system_prompt}_{query}"

    def _build_generate_config(self):
        safety = [
            genai.types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            genai.types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            genai.types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            genai.types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
        ]
        thinking_cfg = None
        if isinstance(self.thinking_budget, int) and self.thinking_budget >= 0:
            thinking_cfg = genai.types.ThinkingConfig(thinking_budget=self.thinking_budget)
        return genai.types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            response_modalities=["TEXT"],
            safety_settings=safety,
            tools=[],  # always none
            thinking_config=thinking_cfg,
            system_instruction=self.system_prompt,
        )

    def query(self, query: str) -> str:
        cache_key = self.get_cache_key(query)
        if cache_key in self.cache:
            return self.cache[cache_key]

        time.sleep(3)
        try:
            if self.model is None:
                return f"Error in {self.model_name} response: client not initialized"

            gen_cfg = self._build_generate_config()
            response = self.model.models.generate_content(
                model=self.model_name,
                contents=query,
                config=gen_cfg
            )

            text = ""
            if getattr(response, "candidates", None):
                cand = response.candidates[0]
                if getattr(cand, "content", None) and getattr(cand.content, "parts", None):
                    text = "".join(
                        getattr(p, "text", "") for p in cand.content.parts if hasattr(p, "text")
                    ).strip()

            if not text:
                text = str(response)

            self.cache[cache_key] = text
            self.save_cache()
            return text

        except Exception as e:
            return f"Error in {self.model_name} response: {e}"

    def delete(self):
        try:
            if getattr(self, "model", None) is not None:
                del self.model
            for attr in ['system_prompt', 'model_name']:
                if hasattr(self, attr):
                    delattr(self, attr)
            gc.collect()
        except Exception as e:
            print(f"Error during deletion of {self.model_name} model: {e}")