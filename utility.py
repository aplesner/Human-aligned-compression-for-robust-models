import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms,datasets
from torchvision.models import ResNet50_Weights
import numpy as np
from kornia.augmentation import RandomJPEG
import os
from src.helpers.utils import load_model
from src.helpers.utils import logger_setup
from default_config import ModelModes
import imageio_ffmpeg as ffmpeg
import imageio
import subprocess
from compressai.zoo import load_state_dict
from ELIC_network import TestModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GetTransforms():
    '''Returns a list of transformations when type as requested amongst train/test
       Transforms('train') = list of transforms to apply on training data
       Transforms('test') = list of transforms to apply on testing data'''

    def __init__(self, dataset_name):
        if dataset_name == 'cifar10':
            self.mean = (0.491, 0.482, 0.446)
            self.std = (0.247, 0.243, 0.261)
            self.crop = False
        if dataset_name == 'imagenette':
            self.mean = (0.485, 0.456, 0.406)
            self.std = (0.229, 0.224, 0.225)
            self.crop = True

    def trainparams(self):
        crop = []
        if self.crop:
            crop = [transforms.Resize((224,224))]
        train_transformations = crop + [ #resises the image so it can be perfect for our model.
            transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
            transforms.RandomRotation((-7,7)),     #Rotates the image to a specified angel
            transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
            transforms.ToTensor(), # comvert the image to tensor so that it can work with torch
            transforms.Normalize(self.mean, self.std) #Normalize all the images
            ]

        return train_transformations

    def testparams(self):
        crop = []
        if self.crop:
            crop = [transforms.Resize((224,224))]
        test_transforms = crop + [
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ]
        return test_transforms

#helper functions
def denorm(batch, mean=(0.491, 0.482, 0.446), std=(0.247, 0.243, 0.261),device = 'cpu'):
  if isinstance(mean, tuple):
    mean = torch.tensor(mean).to(device)
  if isinstance(std, tuple):
    std = torch.tensor(std).to(device)
  return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

def project_perturbation(data_point,p,perturbation):
    """project perturbation to p norm ball"""
    if p == 'l2':
        perturbation = perturbation * min(1, data_point / np.linalg.norm(perturbation.flatten(1)))
    elif p == 'linf':
        perturbation = np.sign(perturbation) * np.minimum(abs(perturbation), data_point)
    return perturbation

class Imagenette_wrapper_old(nn.Module):

    def __init__(self):
        super(Imagenette_wrapper_old, self).__init__()
        self.imagenette_indices = [0,217,482,491,497,566,569,571,574,701]
        self.softmax = torch.nn.LogSoftmax(1)

    def forward(self, x):
        return self.softmax(x[:,self.imagenette_indices])
    
class Imagenette_wrapper(nn.Module):

    def __init__(self):
        super(Imagenette_wrapper, self).__init__()
        self.imagenette_indices = [0,217,482,491,497,566,569,571,574,701]
        self.softmax = torch.nn.LogSoftmax(1)

    def forward(self, x):
        return x[:,self.imagenette_indices]

class Imagenette_wrapper_ext(nn.Module):

    def __init__(self):
        super(Imagenette_wrapper_ext, self).__init__()
        self.imagenette_indices = [0,217,482,491,497,566,569,571,574,701]
        self.softmax = torch.nn.LogSoftmax(1)

    def forward(self, x):
        out = torch.zeros(x.size()[0],11)
        out[:,:10] = x[:,self.imagenette_indices]
        out[:,10]= torch.max(x,dim = 1)[0]
        return self.softmax(out).to(device)
    

#function to load dataset
def load_testset(name):
    """Load the test dataset
    
    Args:
        dataset (str): The name of the dataset to load
    
    Returns:
        testset: The loaded test dataset
    """
    if name == 'imagenette':
        testset = datasets.Imagenette(root='./data', split='val', download=False, transform=ResNet50_Weights.IMAGENET1K_V2.transforms())
    elif name ==  'cifar10':
        testset = datasets.CIFAR10(root='./data/cifar_10', train=False, download=False, transform=transforms.Compose(GetTransforms(name).testparams()))
    elif name == 'imagenet':
        testset = datasets.ImageNet(root='./data', split='val', transform=ResNet50_Weights.IMAGENET1K_V2.transforms())
    elif name == 'imagenet_1000':
        testset = datasets.ImageNet(root='./data', split='val', transform=ResNet50_Weights.IMAGENET1K_V2.transforms())
        try:
            indices = np.load('data/imagenet_1000.npy')
        except OSError:
            indices = np.random.choice(len(testset), 1000, replace=False)
            np.save('data/imagenet_1000.npy',indices)
        testset = torch.utils.data.Subset(testset, indices)
    else:
        print('Dataset not found') 
    return testset

#functions to create the defense
class jpegDefence(nn.Module):

    def __init__(self, q, device= device):
        super(jpegDefence, self).__init__()
        self.device = device
        self.aug = RandomJPEG(jpeg_quality = torch.tensor([q,q]).to(device), p = 1.0, keepdim = True).to(device)

    def forward(self, x):
        x = denorm(x,device = self.device) 
        x= self.aug(x)
        x= transforms.Normalize((0.491, 0.482, 0.446,), (0.247, 0.243, 0.261,))(x)
        return x
    
class jpegDefence_seq(nn.Module):

    def __init__(self,N, q, device= device):
        super(jpegDefence_seq, self).__init__()
        self.N = N
        self.device = device
        self.aug = RandomJPEG(jpeg_quality = torch.tensor([q,q]).to(device), p = 1.0, keepdim = True).to(device)

    def forward(self, x):
        x = denorm(x,device = self.device)
        for _ in range(self.N): 
            x= self.aug(x)
        x= transforms.Normalize((0.491, 0.482, 0.446,), (0.247, 0.243, 0.261,))(x)
        return x


#create HiFiC defense
class HiFiCDefense(nn.Module):

    def __init__(self, weights='hific_low.pt', device= device):

        super(HiFiCDefense, self).__init__()
        self.device = device
        self.logger = logger_setup(logpath=os.path.join('logs/', 'logs'), filepath=os.path.abspath('logs/'))
        self.args,self.transform_model,self.optimizers = load_model('data/'+ weights,self.logger,'cuda',model_mode=ModelModes.EVALUATION)

    def forward(self, x):
        x = denorm(x,device = self.device)
        x,_ = self.transform_model(x)
        x = transforms.Normalize((0.491, 0.482, 0.446,), (0.247, 0.243, 0.261,))(x)
        return x
    
class HiFiCDefense_seq(nn.Module):

    def __init__(self, N, weights='hific_med.pt', device= device):

        super(HiFiCDefense_seq, self).__init__()
        self.N = N
        self.device = device
        self.logger = logger_setup(logpath=os.path.join('logs/', 'logs'), filepath=os.path.abspath('logs/'))
        self.args,self.transform_model,self.optimizers = load_model('data/'+ weights,self.logger,'cuda',model_mode=ModelModes.EVALUATION)

    def forward(self, x):
        x = denorm(x,device = self.device)
        for _ in range(self.N):
            x,_ = self.transform_model(x)
        x = transforms.Normalize((0.491, 0.482, 0.446,), (0.247, 0.243, 0.261,))(x)
        return x
        
class HiFiCDefense_nondiff(nn.Module):

    def __init__(self, weights='data/hific_low.pt', device= device):
        import os
        from src.helpers.utils import load_model
        from src.helpers.utils import logger_setup
        from default_config import ModelModes
        super(HiFiCDefense_nondiff, self).__init__()
        self.device = device
        self.logger = logger_setup(logpath=os.path.join('logs/', 'logs'), filepath=os.path.abspath('logs/'))
        self.args,self.transform_model,self.optimizers = load_model(weights,self.logger,'cuda',model_mode=ModelModes.EVALUATION)
        self.logger.info('Building hyperprior probability tables...')
        self.transform_model.Hyperprior.hyperprior_entropy_model.build_tables()
        self.logger.info('All tables built.')

    def forward(self, x):
        x = denorm(x,device = self.device)
        compression_output = self.transform_model.compress(x,silent=True)
        reconstruction = self.transform_model.decompress(compression_output)
        x = transforms.Normalize((0.491, 0.482, 0.446,), (0.247, 0.243, 0.261,))(reconstruction)
        return x        

class H265Defense(nn.Module):

    def __init__(self, device= device):
        super(H265Defense, self).__init__()
        self.device = device
        self.output_video = 'cache/output_video.mp4'
        self.input_video = 'cache/output_video.mp4'
        self.output_image = 'cache/retrieved_image.png'

    def forward(self, x):
        acc = []
        for xi in x:
            xi = denorm(xi,device = self.device)
            xi = self.float_to_uint8(xi)
            self.encode_h265(xi.squeeze())
            acc.append(self.uint8_to_float(self.decode_h265()))
        x = torch.stack(acc)
        x = transforms.Normalize((0.491, 0.482, 0.446,), (0.247, 0.243, 0.261,))(x)
        return x.to(self.device)
    
    def float_to_uint8(self,image_tensor):
        if not torch.is_floating_point(image_tensor):
            raise ValueError("Input tensor must be of floating-point type")

        # Scale the float values to the range [0, 255]
        image_tensor = image_tensor * 255.0

        # Convert to uint8
        uint8_image_tensor = image_tensor.to(torch.uint8)

        return uint8_image_tensor
    
    def uint8_to_float(self,image_tensor):
        if not image_tensor.dtype == torch.uint8:
            raise ValueError("Input tensor must be of uint8 type")

        # Convert to float
        float_image_tensor = image_tensor.to(torch.float32) / 255.0

        return float_image_tensor

    def encode_h265(self,img):
        # Path to the input image and output video file
        image = img.permute(1, 2, 0).cpu().numpy()  # Convert (C, H, W) to (H, W, C)
        # Get the dimensions of the image

        
        
        # Get the dimensions of the image
        height, width, _ = image.shape
        
        # Construct the FFmpeg command
        command = [
            ffmpeg.get_ffmpeg_exe(),
            '-y',  # Overwrite output file if it exists
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',  # Size of one frame
            '-pix_fmt', 'rgb24',
            '-r', '1',  # Frame rate
            '-i', '-',  # Input comes from a pipe
            '-an',  # No audio
            '-vcodec', 'libx265',  # Use the H.265 codec
            '-crf', '28',  # Quality level (lower is better)
            self.output_video
        ]
        
        # Run FFmpeg command
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process.communicate(input=image.tobytes())

    def decode_h265(self):
        # Path to the input video and output image file


        # Command to extract the first frame from the video
        command = [
            'ffmpeg','-loglevel','error',
            '-i', self.input_video,
            '-vf', 'select=eq(n\\,0)',  # Select the first frame
            '-vsync', 'vfr',
            '-q:v', '2',  # Quality level (lower is better)
            '-y',
            self.output_image
        ]

        # Run FFmpeg command
        subprocess.run(command, check=True)

        # Read the extracted image
        image = imageio.imread(self.output_image)
        return torch.tensor(image).permute(2, 0, 1)

class ELICDefense(nn.Module):

    def __init__(self, weights = '0016',device= device):
        super(ELICDefense, self).__init__()
        self.device = device
        self.state_dict = load_state_dict(torch.load(f'data/ELIC_{weights}_ft_3980_Plateau.pth.tar'))
        self.model = TestModel().from_state_dict(self.state_dict)
        self.model = self.model.to(device)
        self.model.eval()

    def forward(self, x):
        x = denorm(x,device = self.device)
        x = transforms.Resize((256,256))(x)
        x = self.model(x)['x_hat']
        x = transforms.Resize((224,224))(x)
        x = transforms.Normalize((0.491, 0.482, 0.446,), (0.247, 0.243, 0.261,))(x)
        return x

class ELICDefense_seq(nn.Module):

    def __init__(self, N, weights = '0016',device= device):
        super(ELICDefense_seq, self).__init__()
        self.device = device
        self.N = N
        self.state_dict = load_state_dict(torch.load(f'data/ELIC_{weights}_ft_3980_Plateau.pth.tar'))
        self.model = TestModel().from_state_dict(self.state_dict)
        self.model = self.model.to(device)
        self.model.eval()

    def forward(self, x):
        x = denorm(x,device = self.device)
        x = transforms.Resize((256,256))(x)
        for _ in range(self.N):
            x = self.model(x)['x_hat']
        x = transforms.Resize((224,224))(x)
        x = transforms.Normalize((0.491, 0.482, 0.446,), (0.247, 0.243, 0.261,))(x)
        return x

def create_defense(defense, param):
    if defense == 'jpeg':
        return jpegDefence(float(param))
    elif defense == 'HiFiC':
        if param.endswith('.pt'):
            return HiFiCDefense(weights = param)
        return HiFiCDefense()
    elif defense == 'HiFiC_nondiff':
        if param.endswith('.pt'):
            return HiFiCDefense_nondiff(weights = param)
        return HiFiCDefense_nondiff()
    elif defense == 'H265':
        return H265Defense()
    elif defense == 'ELIC':
        if param == '0450' or param == '0004' or param == '0004' or param == '0016' or param == '0032' or param == '0008' or param == '0150':
            return ELICDefense(weights = param)
        return ELICDefense()
    elif defense == 'sequence':
        defenses = param.split(',')
        return torch.nn.Sequential(*[create_defense(*defense.split(':')) for defense in defenses])
    elif defense == 'HiFiC_seq':
        N,weights = param.split(':')
        if weights.endswith('.pt'):
            return HiFiCDefense_seq(int(N),weights = weights)
        return HiFiCDefense_seq(int(N))
    elif defense == 'jpeg_seq':
        N,q = param.split(':')
        return jpegDefence_seq(int(N),float(q))
    elif defense == 'ELIC_seq':
        N,weights = param.split(':')
        if weights == '0450' or weights == '0004':
            return ELICDefense_seq(int(N),weights = weights)
        return ELICDefense_seq(int(N))
    else:
        print('Defense not found')
        return None

