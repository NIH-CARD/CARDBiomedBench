import os
import gc
import json
import shutil
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, TRANSFORMERS_CACHE

class HuggingFaceQuery:
    def __init__(self, system_prompt, model_name, max_tokens, do_sample, torch_dtype=torch.bfloat16):
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.do_sample = do_sample
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        self.cache_file = self.get_cache_file_path()
        self.cache = self.load_cache()
        self.model, self.tokenizer = self._initialize_model_and_tokenizer()

    def _initialize_model_and_tokenizer(self):
        try:
            load_dotenv(os.path.join(os.path.dirname(__file__), '../../configs/.env'))
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map="auto" if torch.cuda.is_available() else None
            )
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            return model, tokenizer
        except Exception as e:
            print(f"Error initializing model and tokenizer for {self.model_name}: {e}")
            return None, None

    def get_cache_file_path(self):
        """
        Get the path to the cache file based on the last part of the model name,
        with a fallback if splitting by '/' is not possible.

        Returns:
        - str: The cache file path.
        """
        try:
            # Try splitting the model name by '/' and get the last part
            model_base_name = self.model_name.split('/')[-1]
        except Exception as e:
            print(f"Error splitting model name {self.model_name}: {e}")
            # Fallback to using the entire model name if splitting fails
            model_base_name = self.model_name

        # Define the cache directory
        cache_dir = os.path.join(os.path.dirname(__file__), '..', '..', '.cache', 'model_responses_cache')
        
        # Ensure the cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # Return the path to the cache file
        return os.path.join(cache_dir, f'{model_base_name}_cache.json')


    def load_cache(self):
        """
        Load the cache from the cache file.
        
        Returns:
        - dict: The loaded cache data.
        """
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading cache file: {e}")
        return {}

    def save_cache(self):
        """
        Save the cache to a cache file.
        """
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            print(f"Error saving cache file: {e}")

    def get_cache_key(self, query: str):
        """
        Generate a unique cache key based on the model name, query, and system prompt.

        Parameters:
        - query (str): The input query string.

        Returns:
        - str: The cache key.
        """
        return f"{self.model_name}_{self.system_prompt}_{query}"

    def query(self, query: str) -> str:
        """
        Query the Hugging Face model or retrieve from cache if available.

        Parameters:
        - query (str): The input query string.

        Returns:
        - str: The generated text (response) or an error message.
        """
        cache_key = self.get_cache_key(query)

        # Check if the result is already cached
        if cache_key in self.cache:
            return self.cache[cache_key]

        # If not cached, query the model
        try:
            if self.model is None or self.tokenizer is None:
                return "Model or tokenizer not initialized."
            
            input_text = self.system_prompt + query
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_tokens,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=self.do_sample,
                    temperature=None if not self.do_sample else 0.6,
                    top_p=None if not self.do_sample else 0.9
                )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove the input text from the generated text
            response_text = generated_text[len(input_text):].strip()

            # Cache the result
            self.cache[cache_key] = response_text
            self.save_cache()

            return response_text
        except Exception as e:
            error_message = f"Error in {self.model_name} response: {e}"
            return error_message

    def delete(self):
        """
        Delete the model and tokenizer to free up memory.
        """
        try:
            if self.model is not None:
                del self.model
                torch.cuda.empty_cache()  # Clear CUDA cache if using GPU

            if self.tokenizer is not None:
                del self.tokenizer

            # Explicitly delete other attributes if necessary
            for attr in ['system_prompt', 'model_name', 'device', 'torch_dtype']:
                if hasattr(self, attr):
                    delattr(self, attr)

            # Clear HuggingFace models cache
            shutil.rmtree(TRANSFORMERS_CACHE)

            # Clear any remaining references
            gc.collect()
        except Exception as e:
            print(f"Error during deletion of model and tokenizer: {e}")