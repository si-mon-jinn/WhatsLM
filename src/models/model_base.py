import json
import os
import pickle

from .models import lookup_table
from .models import FFWD
from .models import FFWD_BOW
from .models import FFWD_BOW_V
from .models import FFWD_BOW_VQK
from .models import MHA



class ModelLoader():
    def __init__(self, model_param_json: str, data_utils = None, device: str = 'cpu'):

        self.params = {}
        self.snap_params = {}

        self.model = None

        self.params = self._load_parameters(model_param_json)

        ######## Check if snapshot exists and load it, otherwise create new model

        self.snaps_json_path = "/".join(model_param_json.split("/")[:-1] + \
                        [self.params['snapshots']['snapshots_path'], ".".join(model_param_json.split("/")[-1].split(".")[:-1])+"_snapshots.json"])
                        
        if os.path.exists(self.snaps_json_path):
            self.snap_params = self._load_parameters(self.snaps_json_path)


            if 'snapshots' in self.snap_params.keys():
                snaps_list = self.snap_params['snapshots']
                if len(snaps_list) > 0 and os.path.exists(self.snap_params['snapshots'][-1]):
                    self.model = self._load_snapshot(self.snap_params['snapshots'][-1])
            
            if self.model == None:
                print('FATAL ERROR: model not loaded correctly')

        else:
            self.model = Model(self.params, data_utils, device)

        ########

    def save(self, file_path: str = ""):
        snapshot_file = "/".join(self.snaps_json_path.split("/")[:-1] + [".".join(self.snaps_json_path.split("/")[-1].split(".")[:-1])+"_"+str(self.model.curr_steps)+".pickle"])

        if self.snap_params == {}: # first save call, create list
            try:
                os.makedirs('/'.join(self.snaps_json_path.split('/')[:-1]))
            except FileExistsError:
                pass

            self.snap_params['snapshots'] = [snapshot_file]
        else: # append to existing list
            self.snap_params['snapshots'].append(snapshot_file)

        self._save_snapshot(snapshot_file)
        self._save_parameters(self.snaps_json_path, self.snap_params)

        return

    def _load_snapshot(self, snap_path: str):
        with open(snap_path, 'rb') as file:
            return pickle.load(file)
        
    
    def _save_snapshot(self, file_path: str):
        with open(file_path, 'bw') as file:
            pickle.dump(self.model, file)
        
    
    def _check_snapshot(self) -> bool:
        return 
    
    def _load_parameters(self, json_path):
        with open(json_path, 'r') as json_file:
            return json.loads(json_file.read())
        return
    
    def _save_parameters(self, json_path: str, params: dict):
        with open(json_path, 'w') as json_file:
            return json_file.write(json.dumps(params))
        return
    



class Model():
    def __init__(self,  model_params: str, data_utils = None, device: str = 'cpu'):
        self.tokenizer = data_utils.tokenizer
        
        self.params = model_params
        self.device = device

        self.model = None

        self.curr_steps = 0
        self.model_loss = []
        self.model_vloss = []

        ######## Checking if I already have info about vocab_size and block_size. 
        # ATM they are info coming from tokenizer and dealer, before the model is created,
        # thus I am not sure their more logical place is in the model parameters...

        if data_utils is not None and 'vocab_size' not in self.params.keys(): 
            self.vocab_size = data_utils.tokenizer.get_vocab_size()
            print(f'taking {self.vocab_size=}')
        if data_utils is not None and 'block_size' not in self.params["model_handler"].keys(): 
            self.block_size = data_utils.dealer.get_block_size()
            print(f'taking {self.block_size=}')
        else:
            self.block_size = self.params["model_handler"]["block_size"]
            
        ########

        self._load_model()
    
    def _load_model(self):       
        model_name = self.params["model_handler"]["model_name"]

        if model_name == "lookup_table":
            vocab_size = self.vocab_size
            block_size = self.block_size

            self.model = lookup_table.lookup_table(vocab_size, block_size).to(self.device)
            print(f'created {model_name} model')

        elif model_name == 'FFWD':
            vocab_size = self.vocab_size
            embed_size = self.params["model_handler"]['embed_size']
            
            self.model = FFWD.FFWD(vocab_size, embed_size).to(self.device)
        
        elif model_name == 'FFWD_BOW':
            vocab_size = self.vocab_size
            embed_size = self.params["model_handler"]['embed_size']
            
            self.model = FFWD_BOW.FFWD_BOW(vocab_size, embed_size, block_size=self.block_size).to(self.device)

        elif model_name == 'FFWD_BOW_V':
            vocab_size = self.vocab_size
            embed_size = self.params["model_handler"]['embed_size']
            
            self.model = FFWD_BOW_V.FFWD_BOW_V(vocab_size, embed_size, block_size=self.block_size).to(self.device)
        
        elif model_name == 'FFWD_BOW_VQK':
            vocab_size = self.vocab_size
            embed_size = self.params["model_handler"]['embed_size']
            head_size = self.params["model_handler"]['head_size']
            
            self.model = FFWD_BOW_VQK.FFWD_VQK(vocab_size, embed_size, head_size, block_size=self.block_size).to(self.device)

        elif model_name == 'MHA':
            vocab_size = self.vocab_size
            embed_size = self.params["model_handler"]['embed_size']
            head_size = self.params["model_handler"]['head_size']
            num_heads = self.params["model_handler"]['num_heads']
            
            self.model = MHA.MHA(vocab_size, embed_size, num_heads, head_size, block_size=self.block_size).to(self.device)
            
        # elif model_name == "GPT":
        #     vocab_size = self.vocab_size
        #     block_size = self.block_size
        #     n_embed = self.params["model_handler"]['n_embed']
        #     n_head_embed = self.params["model_handler"]['n_head_embed']

        #     self.model = GPT.GPT(vocab_size, n_embed, n_head_embed, block_size)
        #     print(f'created {model_name} model')
                
        else:
            print(f'{self.params["model_name"]} is not known')

    def train_model(self, steps: int, trainer):
        trainer.train(steps)
    
    def status(self):
        print(f'Number of steps: {self.curr_steps}')
        print(f'Training loss: {self.model_loss[-1] if len(self.model_loss)>0 else None}')
        print(f'Validation loss: {self.model_vloss[-1] if len(self.model_vloss)>0 else None}')

    def generate(self, prompt: str, tokens: int) -> str:
        return self.tokenizer.decode(self.model.generate(self.tokenizer.encode(prompt).unsqueeze(0).to(self.device), tokens).squeeze())
    
    def get_parameters(self, param = None):
        if param == None:
            return self.params
        else:
            if param in self.params.keys():
                return self.params[param]
            else:
                return f'{param} not in parameters'
        