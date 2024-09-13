import os
from scripts.scripts_utils import load_config

# Load configuration
CONFIG = load_config()

# Template samples
TEMPLATE_SAMPLES = CONFIG.get('template_samples', 1)

# Retrieve system prompt
SYSTEM_PROMPT = CONFIG.get('system_prompt', '')

# Setup models dictionary
MODELS_DICT = {name: details for name, details in CONFIG.get('models', {}).items() if details.get('use', False)}
for model in MODELS_DICT:
    MODELS_DICT[model]['query_instance'] = None

# Retrieve grading model
GRADING_MODEL = CONFIG.get('grading_model', '')

# Load metrics and filter out those not to be used
METRICS_DICT = {name: details for name, details in CONFIG.get('metrics', {}).items() if details.get('use', False)}

# Set max tokens for models
MAX_NEW_TOKENS = CONFIG.get('max_tokens', 1024)

# Set temperature for models
TEMPERATURE = 0

# Set the local .cache directory for huggingface models
HF_HOME = CONFIG['cache_directory']
os.environ['HF_HOME'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '../.cache/huggingface'))