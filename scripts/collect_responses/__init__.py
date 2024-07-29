import os
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '../../configs/.env'))

# Load configuration file
config_path = os.path.join(os.path.dirname(__file__), '../../configs/config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

SYSTEM_PROMPT = config['system_prompt']
MODELS = config['models']