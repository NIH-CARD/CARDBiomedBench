from scripts.utils import load_config

# Load configuration
CONFIG = load_config()

# Retrieve system prompt
SYSTEM_PROMPT = CONFIG.get('system_prompt', '')

# Load models and filter out those not to be used
MODELS = {name: details for name, details in CONFIG.get('models', {}).items() if details.get('use', False)}

# Load metrics and filter out those not to be used
METRICS = {name: details for name, details in CONFIG.get('metrics', {}).items() if details.get('use', False)}
