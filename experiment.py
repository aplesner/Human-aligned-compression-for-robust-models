import torch
from tqdm import tqdm
from utility import denorm,project_perturbation
from torchvision import transforms
import torch.nn.functional as F
import time
import torchattacks
"""
This class is used to run an experiment. It takes a model, a testset and a defense model as input and runs an experiment on the testset.

"""

class Experiment:
    def __init__(self,model_attack,testset,model_defense = None,epsilons = [1/255,2/255,4/255,8/255,10/255,12/255],dataset_name = 'imagenette',device = None,batch_size = 64):
        if dataset_name == 'cifar10':
            self.mean = (0.491, 0.482, 0.446)
            self.std = (0.247, 0.243, 0.261)
        if dataset_name == 'imagenette' or dataset_name == 'imagenet' or dataset_name == 'imagenet_1000':
            self.mean = (0.485, 0.456, 0.406)
            self.std = (0.229, 0.224, 0.225)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available else "cpu"
        else:
            self.device = device
        self.batch_size = batch_size
        self.model_attack = model_attack.to(self.device)
        if model_defense is None:
            self.model_defense = model_attack.to(self.device)
        else:
            self.model_defense = model_defense.to(self.device)
        self.model_attack.eval()
        self.model_defense.eval()
        self.testset = testset
        
        self.predictions_done = False
        
        self.epsilons = epsilons



    def get_universals(self,delta = 0.2,n_images = 100, max_iter = 100000,norm = 'l2', norm_value = 100):
        """ This function is used to create universal adversarial perturbations.
        
        Args:
            delta: the desired fooling rate
            n_images: the number of images to use for
            max_iter: the maximum number of iterations to run
            norm: the norm to use for the perturbation
            norm_value: the maximal value of the norm the perturbation can have
        """
        #prepare attack function and data subset:
        adv_attack_fun = torchattacks.attacks.deepfool.DeepFool(self.model_attack,overshoot = 0.1, steps = 200)
        adv_attack_fun.set_normalization_used(mean=self.mean,std=self.std)
        targets = [s[1] for s in self.testset._samples]
        indices = torch.where(torch.isin(torch.tensor(targets), torch.tensor([0])))[0]
        imgs = torch.utils.data.Subset(self.testset, indices[:n_images])
        test_loader = torch.utils.data.DataLoader(imgs, batch_size=1,
                                                 shuffle=True)
        #initialize variables
        fooling_rate = 0.0
        fooling_rates = []
        fooling_rates.append(fooling_rate)
        iterarion = 0
        u_perturbation = torch.zeros(3,224,224).to(self.device)
        while fooling_rate < 1-delta and iterarion < max_iter:
            iterarion +=1
            
            fooling_rate = 0.0
            for index, (data, target) in tqdm(enumerate(test_loader)):
                #iterate over the images and compute the perturbation
                if index >= n_images:
                    break
                data,target = data.to(self.device),target.to(self.device)
                original_prediction = torch.argmax(self.model_attack(data))
                perturbed_image = data + u_perturbation
                perturbed_prediction = torch.argmax(self.model_attack(perturbed_image))
                if original_prediction == perturbed_prediction:
                    adv_image = adv_attack_fun(perturbed_image,original_prediction.unsqueeze(0))
                    if torch.argmax(self.model_attack(adv_image)) == original_prediction:
                        print('unable to fool model')
                    delta_perturbation = adv_image - perturbed_image
                    u_perturbation = project_perturbation(norm_value,norm,(u_perturbation + delta_perturbation).cpu().detach()).to(self.device)
                else:
                    fooling_rate += 1/n_images
            print(f'iteration {iterarion} fooling rate = {fooling_rate}')
            print(f'l2 norm of perturbation = {torch.norm(u_perturbation,p=2)}')
            fooling_rates.append(fooling_rate)
        print(f'fooling rates = {fooling_rates}')  
        print(f'final fooling rate = {fooling_rate}')
        print(f'final iteration = {iterarion}')
        print(f'l2 norm of perturbation = {torch.norm(u_perturbation,p=2)}')
        return u_perturbation
    
    def test_perturbation(self,perturbation):
        """ This function is used to test a perturbation on the testset."""
        test_loader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size,
                                                 shuffle=False)
        correct = 0
        perturbation = perturbation.to(self.device)
        if self.model_defense is None:
            self.model_defense = self.model_attack
        with torch.no_grad():
            start_time = time.time()
            for index, (data, target) in tqdm(enumerate(test_loader)):
                #prepare data and target
                data,target = data.to(self.device),target.to(self.device)
                #get output prediction
                output = self.model_defense(data+perturbation)
                # Check for success
                for i,o in enumerate(output):
                    if torch.argmax(o)== target[i]: # get the index of the max log-probability
                        correct +=1
            end_time = time.time()
            
        
        final_acc = correct / float(len(self.testset))
        print(f'time taken = {end_time-start_time} for {len(self.testset)} images, or {(end_time-start_time)/len(self.testset)} per image')
        print(f"Test Accuracy = {correct} / {len(self.testset)} = {final_acc}")
        return final_acc
            



    def get_adv_images(self,eps = 8/255,adv_attack ='FGSM',batches=0):
        """ This function is used to generate adversarial images to visualize the results of an attack.
        It returns adversarial images
        """
        test_loader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size,
                                                 shuffle=False)
       #choose attack method
        if adv_attack == 'FGSM':
            adv_attack_fun = torchattacks.attacks.fgsm.FGSM(self.model_attack,eps=eps)
        elif adv_attack == 'iFGSM':
            adv_attack_fun = torchattacks.attacks.bim.BIM(self.model_attack,eps=eps, alpha=eps/4, steps=10)
        elif adv_attack == 'PGD':
            adv_attack_fun =  torchattacks.attacks.pgd.PGD(self.model_attack,eps=eps, alpha=eps/4, steps=10, random_start=True)
        else:
            raise RuntimeError('Unknown attack method')
        if eps == 0:
            print('using vanila')
            adv_attack_fun = torchattacks.attacks.vanila.VANILA(self.model_attack)
        adv_attack_fun.set_normalization_used(mean=self.mean,std=self.std)
        adv_images = []
        print(f'saving {batches*self.batch_size} images')
        for index, (data, target) in tqdm(enumerate(test_loader)):
            #prepare data and target
            data,target = data.to(self.device),target.to(self.device)
            data.requires_grad = True
            #run adversarial attack
            adv_image = adv_attack_fun(data,target)
            adv_image = denorm(adv_image,mean=self.mean,std=self.std ,device = self.device)
            for i in adv_image:
                adv_images.append(i.cpu().detach().numpy())
            print(f'batch {index} done')

            if index >= batches:
                break

        return adv_images



    def run_experiment(self,method = 'FGSM'):
        """ This function is used to run an experiment on the testset."""
        accuracies = []
        examples = []

        test_loader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size,
                                                 shuffle=False)
        #run l2 attacks
        if method == 'DeepFool' or method == 'CW':
            for i in range(len(self.epsilons)):
                self.epsilons[i] = self.epsilons[i] * 255
            accuracies = self.l2_test(test_loader,method)
            return accuracies

        # Run test for each epsilon
        for eps in self.epsilons:
            acc = self.test(test_loader, eps,adv_attack = method)
            accuracies.append(acc)
            

        return accuracies
    
    def copmpute_l2_norm(self,adv_images,original_images):
        l2_norm = torch.zeros(len(adv_images))
        for i in range(len(adv_images)):
            l2_norm[i] = torch.norm(adv_images[i]-original_images[i],p=2)
        return l2_norm
    
    def l2_test(self,test_loader,adv_attack = 'DeepFool'):
        """ This function runs an l2 attack on the testset."""
        self.model_attack = self.model_attack.to(self.device)
        #choose attack method
        if adv_attack == 'DeepFool':
            adv_attack_fun = torchattacks.attacks.deepfool.DeepFool(self.model_attack)        
        elif adv_attack == 'CW':
            adv_attack_fun = torchattacks.attacks.cw.CW(self.model_attack)
        adv_attack_fun.set_normalization_used(mean=self.mean,std=self.std)

        correct = torch.zeros(len(self.epsilons)+1)
        adv_images = []
        targets = []
        l2_norms = []
        print(f'running {adv_attack} attack')
        for index, (data, target) in tqdm(enumerate(test_loader)):
            #prepare data and target
            data,target = data.to(self.device),target.to(self.device)
            data.requires_grad = True
            #run adversarial attack
            adv_image = adv_attack_fun(data,target)
            l2_norm = self.copmpute_l2_norm(adv_image.to(self.device),data)
            for i in adv_image:
                adv_images.append(i.detach().cpu())
            for i in target:
                targets.append(i.detach().cpu().numpy())
            for i in l2_norm:
                l2_norms.append(i.detach().cpu())
        print(f'{adv_attack} attack done')
        adv_data = list(zip(adv_images,targets,l2_norms))
        adv_loader =  torch.utils.data.DataLoader(adv_data, batch_size=self.batch_size,
                                                 shuffle=False)
        if not self.predictions_done:
            self.get_predictions()
        with torch.no_grad():
            for batch_i,(adv_image,target,l2) in tqdm(enumerate(adv_loader),disable=True):
                #get output prediction
                adv_image,target = adv_image.to(self.device),target.to(self.device)
                output = self.model_defense(adv_image)
                non_adv_output = self.predictions[batch_i*self.batch_size:(batch_i+1)*self.batch_size]   
    
                # Check for success
                for i,o in enumerate(output):
                    for j,eps in enumerate(self.epsilons):
                        if l2[i] > eps :
                            o_m = non_adv_output[i]
                        else:
                            o_m = torch.argmax(o)
                        if o_m == target[i]: # get the index of the max log-probability
                            correct[j] +=1
                    if torch.argmax(o)== target[i]: # get the index of the max log-probability                       
                        correct[-1] +=1

        final_acc = [c / float(len(self.testset)) for c in correct]
        print(f'Average l2_norm = {sum(l2_norms)/len(l2_norms)}')
        print(f"{adv_attack} Accuracies = {correct} / {len(self.testset)} = {final_acc}")
        return final_acc[:-1]


    def test(self,test_loader,eps,adv_attack = 'FGSM'):
        """This function runs and adversarial attack for a given epsilon."""
        self.model_attack = self.model_attack.to(self.device)
        #choose attack method
        if adv_attack == 'FGSM':
            adv_attack_fun = torchattacks.attacks.fgsm.FGSM(self.model_attack,eps=eps)
        elif adv_attack == 'iFGSM':
            adv_attack_fun = torchattacks.attacks.bim.BIM(self.model_attack,eps=eps, alpha=eps/4, steps=10)
        elif adv_attack == 'PGD':
            adv_attack_fun =  torchattacks.attacks.pgd.PGD(self.model_attack,eps=eps, alpha=eps/4, steps=10, random_start=True)
        else:
            raise RuntimeError('Unknown attack method')
        adv_attack_fun.set_normalization_used(mean=self.mean,std=self.std)
        correct = 0
        adv_images = []
        targets = []
        print(f'running {adv_attack} for {eps}')
        for index, (data, target) in tqdm(enumerate(test_loader)):
            #prepare data and target
            data,target = data.to(self.device),target.to(self.device)
            data.requires_grad = True
            #run adversarial attack
            adv_image = adv_attack_fun(data,target)
            with torch.no_grad():
                output = self.model_defense(adv_image)
   
                # Check for success
                for i,o in enumerate(output):
                    if torch.argmax(o)== target[i]: # get the index of the max log-probability
                        correct +=1

        final_acc = correct / float(len(self.testset))
        print(f"Epsilon: {eps}\tTest Accuracy = {correct} / {len(self.testset)} = {final_acc}")
        return final_acc
    

    def get_predictions(self):
        """ This function is used to get the predictions of the defense model on the testset without an attack."""
        if hasattr(self, 'base_accuracy'):
            return self.base_accuracy
        predictions = []
        test_loader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size,
                                                 shuffle=False)
        correct = 0
        with torch.no_grad():
            for data, target in tqdm(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model_defense(data)
                for i,o in enumerate(output):
                    if torch.argmax(o)== target[i]: # get the index of the max log-probability
                        correct +=1
                try:
                    predictions.extend(output.argmax(dim=1, keepdim=True).squeeze().detach().cpu().numpy())
                except:
                    #print(data.size())
                    #print(output)
                    predictions.append(output.argmax(dim=1, keepdim=True).squeeze().detach().cpu().numpy().item())

                    pass
            print(f"Defense model base accuracy: {correct/len(self.testset)} ")
        self.predictions = predictions
        self.perdictions_done = True
        self.base_accuracy = correct/len(self.testset)
        return correct/len(self.testset)



   