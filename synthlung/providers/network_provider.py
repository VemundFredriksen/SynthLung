from synthlung.networks.simple_network import SimpleNN

class NetworkProvider():
    def __init__(self, config) -> None:
        if (config["network"] == "SimpleNN"):
            self.network = SimpleNN()

    def __call__(self):
        return self.network