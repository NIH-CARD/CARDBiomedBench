import os
import gc
import time
import json
from dotenv import load_dotenv
from openai import OpenAI

class GPTQuery:
    def __init__(self, system_prompt, model_name, max_tokens):
        self.client = self.initialize_openai_client()
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.max_tokens = max_tokens

    @staticmethod
    def initialize_openai_client():
        """
        Initialize the OpenAI client.

        Returns:
        - OpenAI: Initialized OpenAI client.
        """
        try:
            load_dotenv(os.path.join(os.path.dirname(__file__), '../../configs/.env'))
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if (openai_api_key):
                return OpenAI(api_key=openai_api_key)
            else:
                print("OpenAI API key not found in environment variables.")
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
        return None

    def query(self, query: str) -> str:
        """
        Query the OpenAI API with GPT-4o.

        Parameters:
        - query (str): The input query string.

        Returns:
        - str: The response content from the API or an error message.
        """
        try:
            chat_completion = self.client.chat.completions.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": query}
                ]
            )
            response = chat_completion.choices[0].message.content
            return response
        except Exception as e:
            error_message = f"Error in {self.model_name} response: {e}"
            return error_message
    
    def delete(self):
        """
        Delete the client to free up memory.
        """
        try:
            if self.client is not None:
                del self.client

            for attr in ['system_prompt', 'model_name']:
                if hasattr(self, attr):
                    delattr(self, attr)

            gc.collect()
        except Exception as e:
            print(f"Error during deletion of {self.model_name} client: {e}")
    
    def submit_batch_query(self, batch_file_path: str, metadata: dict = None) -> str:
        """
        Submit a batch query to the OpenAI API.

        Parameters:
        - batch_file_path (str): Path to the .jsonl file containing batch requests.
        - metadata (dict): Optional metadata for the batch job.

        Returns:
        - str: The ID of the submitted batch or an error message.
        """
        try:
            batch_input_file = self.client.files.create(
                file=open(batch_file_path, "rb"),
                purpose="batch"
            )

            batch = self.client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata=metadata
            )

            return batch.id
        except Exception as e:
            error_message = f"Error during batch submission: {e}"
            return {"error": error_message}


    def poll_batch_status(self, batch_id: str, poll_freq: int = 15):
        """
        Poll the status of an ongoing batch job.

        Parameters:
        - batch_id (str): The ID of the batch job.

        Returns:
        - dict: The batch results or an error message.
        """
        try:
            batch_status = None
            start_time = time.time()

            while batch_status != "completed":
                time.sleep(poll_freq)
                elapsed_time = time.time() - start_time
                hours, rem = divmod(elapsed_time, 3600)
                minutes, seconds = divmod(rem, 60)
                time_passed = "{:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds))

                batch_info = self.client.batches.retrieve(batch_id)
                batch_status = batch_info.status
                print(f"Batch Status: {batch_status} | Time Passed: {time_passed}")

                if batch_status == "failed" or batch_status == "expired":
                    return {"error": f"Batch {batch_status} with error"}

            # Retrieve and return the results once completed
            output_file_id = batch_info.output_file_id
            file_response = self.client.files.content(output_file_id)
            batch_results = file_response.text

            return batch_results

        except Exception as e:
            error_message = f"Error during batch polling: {e}"
            return {"error": error_message}

    def cancel_batch(self, batch_id: str):
        """
        Cancel an ongoing batch job.
        
        Parameters:
        - batch_id (str): The ID of the batch job to be canceled.
        
        Returns:
        - dict: The status of the cancellation request.
        """
        try:
            cancellation_response = self.client.batches.cancel(batch_id)
            return cancellation_response
        except Exception as e:
            error_message = f"Error during batch cancellation: {e}"
            return {"error": error_message}