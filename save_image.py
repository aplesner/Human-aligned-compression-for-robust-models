"""This script saves some example adversarial images
"""
import argparse
import json
import yaml
from experiment import Experiment
from net_training import ResNet18
import torch
from utility import load_testset,create_defense, denorm
import pandas as pd
import numpy as np
import datetime
from torchvision.models import resnet50, ResNet50_Weights
import os
from PIL import Image


def load_config(path):
    """Load the configuration from a yaml or json file"""
    with open(path, 'r') as file:
        if path.endswith('.yaml'):
            config = yaml.load(file, Loader=yaml.FullLoader)
        elif path.endswith('.json'):
            config = json.load(file)
        else:
            raise ValueError("Unsupported file format. Use 'yaml' or 'json'.")
    return config

def assign_args_from_config(args, config):
    """Assign the args the corresponding value from the config"""
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            raise ValueError(f"Argument {key} not found in args")

def load_model(model_name):
    if model_name == 'ResNet18':
        model = ResNet18()
        model.load_state_dict((torch.load('./data/resnet18_cifar10_weights.pt',weights_only=True)))
        model.eval()
    if model_name == 'ResNet50':
        model = resnet50()
        model = torch.nn.Sequential(model,torch.nn.Linear(1000,10),torch.nn.LogSoftmax(1))
        model.load_state_dict((torch.load('./data/resnet50_imagenette_weights.pt',weights_only=True)))
        model.eval()
    return model


def combine_models(defense,model):
    return torch.nn.Sequential(defense,model)




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run an adversarial experiment')
    #arguments to load and save congiguration
    parser.add_argument('--use_config', action='store_true', help='Use the config file')
    parser.add_argument('--config', type=str, help='Path to the config file', default='')
    parser.add_argument('--save_config',action='store_true', help='Save the config file')
    #arguments for the experiment
    parser.add_argument('--model_attack', type=str, help='Model used to create the adversarial example', default='ResNet50')
    parser.add_argument('--batch_size', type=int, help='Batch size', default=16)
    parser.add_argument('--dataset', type=str, help='Dataset to use', default='imagenette')
    #arguments for the attack
    parser.add_argument('--attack', type=str, help='The attack to use', default='FGSM')
    parser.add_argument('--norm', type=str, help='Norm to use', default='l_inf')
    parser.add_argument('--epsilons',nargs='*', type=str, help='Epsilon for the attack', default=['8/255'])
    #arguments for the defense
    parser.add_argument('--defense', type=str, help='The defense to use', default=None)
    parser.add_argument('--defense_param', type=str, help='Parameters to use for the defense (like quality for JPEG)', default='25.0')
    parser.add_argument('--non_diff',action='store_true', help='If set the defense uses the non differentiable version')
    parser.add_argument('--attack_through',action='store_true', help='If true the gradient is run through the defense when creating adversarial examples')
    #arguments for the output
    parser.add_argument('--output', type=str, help='Path to save the results', default='results/')
    parser.add_argument('--get_baseline',action='store_true', help='If True the result includes the baseline accuracy')
    args = parser.parse_args()
    print('starting')
    epsilons = [float(n)/float(d) for [n,d] in [str_eps.split('/') for str_eps in args.epsilons]]
    # Load the configuration if use_config is True
    if args.use_config:
        config = load_config(args.config)
        assign_args_from_config(args, config)
        epsilons = [0] + [float(n)/float(d) for [n,d] in [str_eps.split('/') for str_eps in args.epsilons[1:]]]

    args.batch_size = 8
        

    
    # Load the model
    model = load_model(args.model_attack)
    
    # Create the testset
    testset = load_testset(args.dataset)
    # Create the attack model
    file_suffix = ''
    if args.defense is not None:
        # Create the defense
        defense = create_defense(args.defense,args.defense_param)
        model_defense = combine_models(defense,model)
        file_suffix += args.defense
    else:
        model_defense = None

    if args.attack_through:
        model_attack = model_defense
        file_suffix += '_through'
    else:
        model_attack = model

    # Create the experiment
    experiment = Experiment(model_attack=model_attack, testset=testset, model_defense=model_defense,dataset_name=args.dataset, epsilons=epsilons, batch_size=args.batch_size)
    
    
    
   
    # Get example images
    file_name='images/'+ args.dataset + '_' + args.attack + '_' + file_suffix
    names = [n for [n,_] in [str_eps.split('/') for str_eps in args.epsilons]]    

    for name,eps in zip(names,epsilons):
        os.makedirs(f'{file_name}/{name}', exist_ok = True)
        imgs = experiment.get_adv_images(eps,adv_attack = args.attack,batches=1)
        #print(imgs)
        for i,img in enumerate(imgs):
            im = Image.fromarray((img*255).astype(np.uint8).transpose(1,2,0))
            im.save(f'{file_name}/{name}/{i}.png')
    
    dataloader = torch.utils.data.DataLoader(testset,batch_size=8,shuffle=False)
    X,y = next(iter(dataloader))
    imgs = (denorm(defense(X.to('cuda')),device = 'cuda')).detach().cpu().numpy()
    for i,img in enumerate(imgs):
            im = Image.fromarray((img*255).astype(np.uint8).transpose(1,2,0))
            im.save(f'{file_name}/{i}.png')
