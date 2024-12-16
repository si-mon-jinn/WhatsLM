class DataLoader():
    def __init__(self):
        self.raw_data = ""
        return
    
    def _load_raw_data(seldf): return



class txtLoader(DataLoader):
    def __init__(self, txt_fname):
        super().__init__()

        self.raw_data = self._load_raw_data(txt_fname)

        return
    
    def _load_raw_data(self, txt_fname):
        raw_data = open(txt_fname, 'r').read()

        return raw_data
    
