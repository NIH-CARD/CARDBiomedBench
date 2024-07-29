import os
from openai import OpenAI
from . import SYSTEM_PROMPT

openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def query_gpt4o(query: str) -> str:
    """
    Query the OpenAI API using GPT-4o.

    Parameters:
    - query (str): The input query string.

    Returns:
    - str: The response content from the API or an error message.
    """
    try:
        chat_completion = openai_client.chat.completions.create(
            model='gpt-4o',
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ]
        )
        response = chat_completion.choices[0].message.content
        return response
    except Exception as e:
        error_message = f"Error in GPT-4o response: {e}"
        return error_message