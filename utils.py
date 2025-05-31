import yaml
import hashlib

def load_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)

def compute_sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()