""""""

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data

class gan_data:
    """"""
    
    def __init__(self, hpar, norm = None, bno_aug = False):
        self.hpar = hpar
        self.norm = norm
        
        trans = [
           transforms.Resize(self.hpar["image_size"]),
           transforms.CenterCrop(self.hpar["image_size"]),
           transforms.ToTensor(),
           transforms.RandomHorizontalFlip(),
        ]
        
        if bno_aug:
            trans.append(transforms.Pad(padding=10, padding_mode="reflect"))
            trans.append(transforms.RandomAffine(degrees=15., shear=0.1))
            trans.append(transforms.Pad(padding=-10))
            trans.append(transforms.RandomResizedCrop(size=self.hpar["image_size"], scale=(0.85, 1.0)))
            trans.append(transforms.RandomApply([transforms.ColorJitter(0.1, 0.1)], p=0.3))
            
        trans.append(transforms.Grayscale())
        if self.norm is not None:
            trans.append(transforms.Normalize(*self.norm))
        
        
        self.dataset = dset.ImageFolder(
            root=self.hpar["dataroot"],
            transform=transforms.Compose(trans)
        )
        
    def __call__(self):
        return self.dataset


class gan_dataloader:
    """"""
    
    def __init__(self, dataset, hpar):
        self.dataset = dataset
        self.hpar = hpar

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, 
            batch_size=self.hpar["batch_size"],
            shuffle=True,
            num_workers=self.hpar["workers"]
        )
    
    def __call__(self):
        return self.dataloader