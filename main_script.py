"""main script

This main script runs an experiment given a config. It creates the models, runs the attacks and saves the results.

Config parameters:
    model_attack: model used to create the adversarial example
    defense: the defense to use, if not specified no defense will be used.
    attack_through: if true the gradient is run through the defense when creating adversarial examples
    defense_param: parameters to use for the defense (like qality for JPEG)
    output: path to save the results
    dataset: dataset to use
    batch_size: batch size
    epsilons: epsilon values to use for the attack
    attack: the attack to use




"""
import argparse
import json
import yaml
from experiment import Experiment
from net_training import ResNet18
import torch
from utility import load_testset,create_defense, Imagenette_wrapper
import pandas as pd
import numpy as np
import datetime
from torchvision.models import resnet50, ResNet50_Weights,vit_b_16, ViT_B_16_Weights
import os


def save_config(args, path):
    """Save the configuration to a yaml or json file"""
    with open(path, 'w') as file:
        if path.endswith('.yaml'):
            yaml.dump(vars(args), file)
        elif path.endswith('.json'):
            json.dump(vars(args), file,indent=6)
        else:
            raise ValueError("Unsupported file format. Use 'yaml' or 'json'.")


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

def load_model(model_name,dataset):
    if model_name == 'ResNet18':
        model = ResNet18()
        model.load_state_dict((torch.load('./data/resnet18_cifar10_weights.pt',weights_only=True)))
        model.eval()
    elif model_name == 'ResNet50':
        model = resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)
        if dataset == 'imagenette':
            model = torch.nn.Sequential(model,Imagenette_wrapper())
        model.eval()
    elif model_name == 'ResNet50finetuned':
        model = resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)
        model.fc = torch.nn.Linear(2048,10,  bias=True)
        model = torch.nn.Sequential(model,torch.nn.LogSoftmax(1))
        model.load_state_dict((torch.load('./data/resnet50_imagenette_weights_jpeg.pt',weights_only=True)))
        model.eval()
    elif model_name == 'Vit':
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=weights)
        if dataset == 'imagenette':
            model = torch.nn.Sequential(model,Imagenette_wrapper())
        model.eval()
    else:
        raise ValueError('Unknown model')
    return model


def combine_models(defense,model):
    return torch.nn.Sequential(defense,model)

def save_results(results,epsilons, path):
    column_names = epsilons
    results = np.atleast_2d(np.array(results))
    results = pd.DataFrame(results,columns=column_names)
    results.to_csv(path+'.csv')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run an adversarial experiment')
    #arguments to load and save congiguration
    parser.add_argument('--use_config', action='store_true', help='Use the config file')
    parser.add_argument('--config', type=str, help='Path to the config file', default='')
    parser.add_argument('--save_config',action='store_true', help='Save the config file')
    #arguments for the experiment
    parser.add_argument('--model_attack', type=str, help='Model used to create the adversarial example, ResNet50 or Vit', default='ResNet50')
    parser.add_argument('--batch_size', type=int, help='Batch size', default=16)
    parser.add_argument('--dataset', type=str, help='Dataset to use. imagenette,imagenet,imagenet_1000', default='imagenette')
    #arguments for the attack
    parser.add_argument('--attack', type=str, help='The attack to use. FGSM,iFGSM,PGD,CW,DeepFool', default='FGSM')
    parser.add_argument('--norm', type=str, help='Norm to use. This parameter is not used atm', default='l_inf')
    parser.add_argument('--epsilons',nargs='*', type=str, help='Epsilons for the attack in the form X/255. for CW and DeepFool the /255 part is ignored', default=['8/255'])
    #arguments for the defense
    parser.add_argument('--defense', type=str, help='The defense to use. jpeg,HiFiC,ELIC,sequence,jpeg_seq,HiFiC_seq,ELIC_seq', default=None)
    parser.add_argument('--defense_param', type=str, help='Parameters to use for the defense (like quality for JPEG). look at utility.py for more information', default='25.0')
    parser.add_argument('--non_diff',action='store_true', help='If set the defense uses the non differentiable version. only available for HiFiC')
    parser.add_argument('--attack_through',action='store_true', help='If true the gradient is run through the defense when creating adversarial examples')
    #arguments for the output
    parser.add_argument('--output', type=str, help='Path to save the results', default='results/')
    parser.add_argument('--get_baseline',action='store_true', help='If True the result includes the baseline accuracy')
    
    args = parser.parse_args()

    if args.attack_through and args.defense is None:
        print('Cannot attack through  without a defense')
        args.attack_through = False

    #compute epsilons from string input
    epsilons = [float(n)/float(d) for [n,d] in [str_eps.split('/') for str_eps in args.epsilons]]


    # Load the configuration if use_config is True
    if args.use_config:
        config = load_config(args.config)
        assign_args_from_config(args, config)


    # Load the model
    model = load_model(args.model_attack,args.dataset)
    
    # Create the testset
    testset = load_testset(args.dataset)

    # Create the models for attack and defense
    file_suffix = ''
    if args.defense is not None:
        # Create the defense model
        if args.non_diff:
            defense = create_defense(args.defense+'_nondiff',args.defense_param)
        else:
            defense = create_defense(args.defense,args.defense_param)
        model_defense = combine_models(defense,model)
        file_suffix += args.defense
    else:
        model_defense = None

    if args.attack_through:
        #create the attack model
        if args.non_diff:
            defense_diff = create_defense(args.defense,args.defense_param)
            model_attack = combine_models(defense_diff,model)
        else:
            model_attack = model_defense
        file_suffix += '_through'
    else:
        model_attack = model

    # Create the experiment
    experiment = Experiment(model_attack=model_attack, testset=testset, model_defense=model_defense,dataset_name=args.dataset, epsilons=epsilons, batch_size=args.batch_size)
    
    
    
   
    # Run the experiment
    results = experiment.run_experiment(method=args.attack)
    # Save the results
    time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    file_name=args.dataset + '_' + args.attack + '_' + file_suffix + time
        # save the configuration if save_config is True
    if args.save_config:
        if args.config == '':
            args.config = 'configs/'+file_name+'.json'
        save_config(args,args.config)
    # Get baseline accuracy
    if args.get_baseline:
        baseline_accuracy = experiment.get_predictions()
        results = [baseline_accuracy] + results
        args.epsilons = ['0']+args.epsilons
    save_results(results, args.epsilons, args.output+file_name)