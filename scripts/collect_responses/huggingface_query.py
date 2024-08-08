from transformers import AutoModelForCausalLM, AutoTokenizer, TRANSFORMERS_CACHE
from dotenv import load_dotenv
import shutil
import torch
import os
import gc

class HuggingFaceQuery:
    def __init__(self, system_prompt, model_name):
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tokenizer = self._initialize_model_and_tokenizer()

    def _initialize_model_and_tokenizer(self):
        try:
            load_dotenv(os.path.join(os.path.dirname(__file__), '../../configs/.env'))
            model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            return model, tokenizer
        except Exception as e:
            print(f"Error initializing model and tokenizer for {self.model_name}: {e}")
            return None, None

    def query(self, query: str) -> str:
        """
        Query the Hugging Face model.

        Parameters:
        - query (str): The input query string.

        Returns:
        - str: The generated text or an error message.
        """
        try:
            if self.model is None or self.tokenizer is None:
                return "Model or tokenizer not initialized."
            
            input_text = self.system_prompt + query
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=512, num_return_sequences=1)
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text.strip()
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
            for attr in ['system_prompt', 'model_name', 'device']:
                if hasattr(self, attr):
                    delattr(self, attr)

            # Clear huggingface cache
            shutil.rmtree(TRANSFORMERS_CACHE)
            
            # Clear any remaining references
            gc.collect()
        except Exception as e:
            print(f"Error during deletion of {self.model_name} model and tokenizer: {e}")