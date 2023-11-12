import requests
import json

from synthlung.action_provider.action_provider import ActionProvider

class LogRemoteProvider(ActionProvider):
    def __init__(self):
        self.start_session_url = "http://localhost:5167/api/Session/newSession"
        self.new_log_url = "http://localhost:5167/api/Log"
        self.get_config_url = "http://localhost:5167/api/Config/000000000000000000000000"
        self.header = {"Content-Type" : "application/json"}
        self.started = False
        self.verify = False

    def start_new_session(self):
        response = requests.get(self.start_session_url, headers=self.header, verify=self.verify)

        if (response.status_code == 200):
            response_data = response.json()
            self.started = True
            self.session_id = response_data["id"]

    def get_session_id(self):
        if (self.started):
            return self.session_id
        
        print("Session has not started yet")
        exit(2)

    def get_config(self):
        response = requests.get(self.get_config_url, headers=self.header, verify=self.verify)

        if (response.status_code == 200):
            return response.json()

        print("Could not fetch remote config")
        exit(3)
    
    def save_log(self, log_dict):
        if (not self.started):
            return
        
        session_id_dict = {"Session_Id": self.session_id}
        log_dict = {**log_dict, **session_id_dict}
        requests.post(self.new_log_url, data=json.dumps(log_dict), headers=self.header, verify=self.verify)


if (__name__ == "__main__"):
    r = LogRemoteProvider()
    r.start_new_session()
    print(r.get_session_id())
