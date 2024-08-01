import os
from scripts.utils import load_config

# Load configuration
CONFIG = load_config()

# Retrieve system prompt
SYSTEM_PROMPT = CONFIG.get('system_prompt', '')

# Load models and filter out those not to be used
MODELS_DICT = {name: details for name, details in CONFIG.get('models', {}).items() if details.get('use', False)}

# Load metrics and filter out those not to be used
METRICS_DICT = {name: details for name, details in CONFIG.get('metrics', {}).items() if details.get('use', False)}

# Set number of workers for any parallel tasks
PARALLELISM_CONFIG = CONFIG.get('parallelism', {})
NUM_WORKERS = PARALLELISM_CONFIG.get('workers') or os.cpu_count()