from scripts import SYSTEM_PROMPT, MODELS_DICT
from .gpt4o_query import GPT4OQuery
from .gemini_query import GeminiQuery

# Initialize clients/models and create instances of query classes
if 'gpt-4o' in MODELS_DICT:
    MODELS_DICT['gpt-4o']['query_instance'] = GPT4OQuery(SYSTEM_PROMPT)

if 'gemini-1.5-pro' in MODELS_DICT:
    MODELS_DICT['gemini-1.5-pro']['query_instance'] = GeminiQuery(SYSTEM_PROMPT)