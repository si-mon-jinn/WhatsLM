import argparse
import torch

description = \
"""
What's LM? Let's find out. 
"""

parser = argparse.ArgumentParser(description=description)
parser.add_argument("-j", "--json", type=str, help="input file path in json format for model project", required=True)
parser.add_argument("-t", "--train", type=int, help="number of training steps to perform")
parser.add_argument("-g", "--generate", type=int, help="number tokens to generate")
parser.add_argument("-v", "--verbose", action="store_true", help="random prints on what's going on", default=False)
parser.add_argument("-d", "--device", type=str, help="device to train and generate on")
parser.add_argument("-p", "--plot", type=str, help="plot train and validation loss", nargs='?', default='')


args = parser.parse_args()

_train = False
_gen = False
_plot = False

if args.train is not None:
    _train = True
    _training_steps = args.train

if args.generate is not None:
    _gen = True
    _gen_toks = args.generate

# if not _train and not _gen:
#     _gen = True
#     _gen_toks = 50

if args.plot is not None:
    _plot = True
    _snaplot = args.plot


_verbose = True if args.verbose else False

'''
Device could be 
- 'cuda'
- 'cuda:int1:int2' --> int2 is number of cpu cores, int1 gpu id
- 'cpu'
- 'cpu:int' --> int is number of cores
'''

device = 'cuda'
if args.device is not None:
    if args.device.startswith('cuda'):
        device = args.device.split(":")[0]
        if len(args.device.split(":"))>1: torch.set_num_threads(int(args.device.split(':')[1]))
    elif args.device.startswith('cpu'):
        device = 'cpu'
        if len(args.device.split(":"))>1: torch.set_num_threads(int(args.device.split(':')[1]))



if _train or _gen:
    from .src.data import data_handler

    json_input = args.json 

    if _verbose: print('\n>>> Setting up data stuff...')

    data_utils = data_handler.DataHandler(json_input)

    #tokenizer = data_utils.tokenizer

    # Create/Load model -> ModelLoader
    ## Input: model configuration file/model parameters
    ## Output: instance of model class to call forward and stuff during training

    from .src.models.model_base import ModelLoader

    if _verbose: print('\n>>> Setting up model...')

    model_loader = ModelLoader(json_input, data_utils, device=device)

    model = model_loader.model

    if _verbose:
        print('\n>>> Model name:',model.get_parameters()['model_handler']['model_name'])
        print('>>> Parameters dictionary:\n', model.get_parameters())

# Train model -> ModelTrainer
    # Save snapshot every X epochs
# Finalize training -> ModelTrainer
## Input: training configuration file as input or from model configuration file
## Output: model instance after training ready to be used or save it on disk (also statistics on performance)

if _train:
    if _verbose: print('\n>>> Setting up training...')

    from .src.training.trainer import ModelTrainer

    dealer = data_utils.dealer
    trainer = ModelTrainer(model_loader, dealer, json_input, device=device)#, dealer)

    if _gen:
        if _verbose: print('\n>>> Status and generation...')
        model.status() # Terrible, should return a data structure like a dict
        print(model.generate("sucare", _gen_toks))

    model.train_model(_training_steps, trainer)

if _gen:
    if _verbose: print('\n>>> Status and generation...')
    model.status()
    print(model.generate("sucare", _gen_toks))

if _plot:
    from matplotlib import pyplot as plt

    if _snaplot == '' and (_train or _gen):
        print('model')
        loss = model.model_loss[20:]
        vloss = model.model_vloss[20:]

    else:
        print('pickle')
        import pickle

        model = pickle.load(open(_snaplot, 'rb'))
        loss = model.model_loss[20:]
        vloss = model.model_vloss[20:]

    fig, ax = plt.subplots()

    ax.plot(loss, label='train')
    ax.plot(vloss, label='valid')
    ax.legend()
    fig.savefig(model.params['model_handler']['model_name']+"_loss.png")
    #plt.show()
