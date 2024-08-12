from transformers import AutoModelForCausalLM, AutoTokenizer, TRANSFORMERS_CACHE
from dotenv import load_dotenv
import shutil
import torch
import os
import gc

class HuggingFaceQuery:
    def __init__(self, system_prompt, model_name, max_tokens, torch_dtype=torch.float16):
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
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

    def query(self, query: str) -> str:
        """
        Query the Hugging Face model.

        Parameters:
        - query (str): The input query string.

        Returns:
        - str: The generated text (response) or an error message.
        """
        try:
            if self.model is None or self.tokenizer is None:
                return "Model or tokenizer not initialized."
            
            input_text = self.system_prompt + query
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove the input text from the generated text
            response_text = generated_text[len(input_text):].strip()

            return response_text
        except Exception as e:
            error_message = f"Error in {self.model_name} response: {e}"
            return error_message

    def is_on_gpu(self):
        """
        Check if the model and inputs are on the GPU.

        Returns:
        - bool: True if the model is on GPU, otherwise False.
        """
        return next(self.model.parameters()).is_cuda

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

            # Clear huggingface models cache
            shutil.rmtree(TRANSFORMERS_CACHE)
            
            # Clear any remaining references
            gc.collect()
        except Exception as e:
            print(f"Error during deletion of {self.model_name} model and tokenizer: {e}")