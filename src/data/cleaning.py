# Functions to clean raw data (rare words, emoticons, timestamps, ...)

from .loading import DataLoader

class Cleaner():
    def __init__(self): 
        self.clean_data = ""
        
        return

    def _clean(self): return

    def _check(self): return


class txtCleaner(Cleaner):
    def __init__(self, data: DataLoader):
        super().__init__()

        self.clean_data = self._clean(data.raw_data)

    def _clean(self, raw_data):
        allowed_chars = ' 0123456789:-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\n'
        clean_data = ''.join([char for char in raw_data if char in allowed_chars])

        #clean_data = "\n".join(["".join(line.split("-")[1:]) for line in clean_data.split("\n")])

        return clean_data



