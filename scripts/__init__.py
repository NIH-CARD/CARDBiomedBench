import os
from scripts.scripts_utils import load_config

# Load configuration
CONFIG = load_config()

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

# Set number of workers for any parallel tasks
PARALLELISM_CONFIG = CONFIG.get('parallelism', {})
NUM_WORKERS = PARALLELISM_CONFIG.get('workers') or os.cpu_count()