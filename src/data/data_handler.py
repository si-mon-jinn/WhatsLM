import json

from . import loading
from . import cleaning
from . import tokenizing
from . import dealing

class DataHandler():
    def __init__(self, config_path: str):
        self.params = self._load_parameters(config_path)

        self.loader = None
        self.cleaner = None
        self.tokenizer = None
        self.dealer = None

        if self.params['raw_data_format'] == 'single_txt':
            self.loader = loading.txtLoader(self.params['data_path'])
            self.cleaner = cleaning.txtCleaner(self.loader)

            self.tokenizer = tokenizing.Tokenizer(self.cleaner.clean_data)
            self.dealer = dealing.txtDealer(self.tokenizer.encode(self.cleaner.clean_data))
        else:
            print(f'{self.params["raw_data_format"]} not supported.')


    def _load_parameters(self, config_path: str) -> dict[str, any]:
        with open(config_path, 'r') as json_file:
            return json.loads(json_file.read())["data_handler"]
