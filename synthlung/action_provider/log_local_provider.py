import json

from synthlung.action_provider.action_provider import ActionProvider

class LogLocalProvider(ActionProvider):
    def __init__(self):
        pass

    def start_new_session(self):
        pass

    def get_session_id(self):
        return 0

    def get_config(self):
        config_path = "./synthlung/config.json"
        with open(config_path, "r") as f:
            data = json.load(f)
        return data
    
    def save_log(self, log_dict):
        pass
