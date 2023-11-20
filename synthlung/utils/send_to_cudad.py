class SendToCudad(object):
    def __init__(self, keys) -> None:
        self.keys = keys

    def __call__(self, sample: dict) -> dict:
        for key in self.keys:
            sample[key] = sample[key].to('cuda')

        return sample
