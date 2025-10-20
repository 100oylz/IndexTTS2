import os
from huggingface_hub import hf_hub_download


def load_custom_model_from_hf(repo_id, model_filename="pytorch_model.bin", config_filename="config.yml"):
    os.makedirs("./checkpoints", exist_ok=True)
    model_path = hf_hub_download(repo_id=repo_id, filename=model_filename, cache_dir="./checkpoints")
    if config_filename is None:
        return model_path
    config_path = hf_hub_download(repo_id=repo_id, filename=config_filename, cache_dir="./checkpoints")

    return model_path, config_path

def get_lowercase_keys_config(config):
    if not isinstance(config, dict):
        return config
    return {k.lower(): get_lowercase_keys_config(v) for k, v in config.items()}


def override_config(base_config, new_config):
    result = base_config.copy()
    for key, value in new_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = override_config(result[key], value)  # Recursive merge for nested dicts
        else:
            result[key] = value
    return result