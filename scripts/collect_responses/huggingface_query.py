from transformers import pipeline
import torch
from dotenv import load_dotenv
import os
import gc

class HuggingFaceQuery:
    def __init__(self, system_prompt, model_name, task="text-generation"):
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.task = task
        self.pipeline = self._initialize_pipeline()

    def _initialize_pipeline(self):
        try:
            load_dotenv(os.path.join(os.path.dirname(__file__), '../../configs/.env'))
            return pipeline(self.task, model=self.model_name, device=0 if torch.cuda.is_available() else -1)
        except Exception as e:
            print(f"Error initializing pipeline for {self.model_name}: {e}")
            return None

    def query(self, query: str) -> str:
        """
        Query the Hugging Face model.

        Parameters:
        - prompt (str): The input prompt string.

        Returns:
        - str: The generated text or an error message.
        """
        try:
            if self.pipeline is None:
                return "Pipeline not initialized."
            messages = [
                {"role": "user", "content": self.system_prompt + query},
            ]
            result = self.pipeline(messages, max_new_tokens=512)
            return result[0]['generated_text'][-1]["content"].strip()
        except Exception as e:
            error_message = f"Error in {self.model_name} response: {e}"
            return error_message

    def delete(self):
        """
        Delete the pipeline to free up memory.
        """
        try:
            if self.pipeline is not None:
                del self.pipeline
                torch.cuda.empty_cache()  # Clear CUDA cache if using GPU

            # Explicitly delete other attributes if necessary
            for attr in ['system_prompt', 'model_name', 'task']:
                if hasattr(self, attr):
                    delattr(self, attr)

            # Clear any remaining references
            gc.collect()
            print(f"{self.model_name} pipeline and attributes deleted successfully.")
        except Exception as e:
            print(f"Error during deletion of {self.model_name} pipeline: {e}")