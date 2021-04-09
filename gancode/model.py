""""""

import pickle
import torch
import torch.nn as nn
        
        
class Generator(nn.Module):
    """Generator code"""

    def __init__(self, hpar, ngpu):
        super(Generator, self).__init__()
        self.hpar = hpar
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.hpar["nz"], self.hpar["ngf"] * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.hpar["ngf"] * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.hpar["ngf"] * 8, self.hpar["ngf"] * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hpar["ngf"] * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.hpar["ngf"] * 4, self.hpar["ngf"] * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hpar["ngf"] * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.hpar["ngf"] * 2, self.hpar["ngf"], 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hpar["ngf"]),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(self.hpar["ngf"], self.hpar["nc"], 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
        
        
class Discriminator(nn.Module):
    """Discriminator code"""
    def __init__(self, hpar, ngpu):
        super(Discriminator, self).__init__()
        self.hpar = hpar
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.hpar["nc"], self.hpar["ndf"], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.hpar["ndf"], self.hpar["ndf"] * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hpar["ndf"] * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.hpar["ndf"] * 2, self.hpar["ndf"] * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hpar["ndf"] * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.hpar["ndf"] * 4, self.hpar["ndf"] * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.hpar["ndf"] * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.hpar["ndf"] * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class FrogGan:
    """"""
    
    def __init__(self, hpar, ngpu, device, bnew_net: bool = True, train_label: str = "tmp"):
        self.hpar = hpar
        self.ngpu = ngpu
        self.device = device
        self.train_label = train_label
        
        self.netG = Generator(self.hpar, self.ngpu).to(self.device)
        self.netD = Discriminator(self.hpar, self.ngpu).to(self.device)
        
        # Handle multi-gpu if desired
        if (self.device.type == 'cuda') and (self.ngpu > 1):
            self.netG = nn.DataParallel(self.netG, list(range(self.ngpu)))
            self.netD = nn.DataParallel(self.netD, list(range(self.ngpu)))
            
        if bnew_net:
          self.netG.apply(self.weights_init)
          self.netD.apply(self.weights_init)
        elif False:
          self.load_models(train_label)
        else:
          self.load_models("tmp", ".")
          with open('run_infos.p', 'rb') as input:
            self.run_infos = pickle.load(input)
            
    def load_models(label = None, outdir = None):
        """Import generator and discriminator state dictionaries"""
    
        if outdir is None:
            outdir = "model_backups"
        if label is None:
            label = self.train_label
    
        self.netG.load_state_dict(torch.load(f"{outdir}/netG_state_dict_{label}.pt"))
        self.netD.load_state_dict(torch.load(f"{outdir}/netD_state_dict_{label}.pt"))
        self.netG.eval()
        self.netD.eval()
    
    def save_models(label = None, outdir = None):
        """Write generator and discriminator state dictionaries"""
    
        if outdir is None:
            outdir = "model_backups"
        if label is None:
            label = self.train_label
    
        torch.save(self.netG.state_dict(), f"{outdir}/netG_state_dict_{label}.pt")
        torch.save(self.netD.state_dict(), f"{outdir}/netD_state_dict_{label}.pt")
    
    
    @staticmethod
    def weights_init(m):
        """Custom weights initialization for netG and netD
        
        Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
        """
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    