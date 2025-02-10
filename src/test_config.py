from config import load_config, Config

config = load_config("config.yaml")
print(f"Type of config.model_ensemble: {type(config.model_ensemble)}")
print(f"Does config.model_ensemble have 'keys' attribute? : {'keys' in dir(config.model_ensemble)}") # Check if 'keys' is in dir
print(f"Trying to call config.model_ensemble.keys():")
try:
    keys = config.model_ensemble.keys()
    print(f"Successfully got keys: {keys}")
except AttributeError as e:
    print(f"AttributeError: {e}")

    