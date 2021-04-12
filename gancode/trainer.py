""""""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from torchvision.utils import save_image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from pytorch_fid.fid_score import calculate_fid_given_paths


from PIL import Image, ImageEnhance
from torchvision import transforms


class GanTrainer:
    """"""
    def __init__(self, gan_net, dataloader, h_par_train: dict):
        self.gan_net = gan_net
        self.netD = self.gan_net.netD
        self.netG = self.gan_net.netG
        self.h_par_model = self.gan_net.hpar
        self.device = self.gan_net.device
        self.dataloader = dataloader
        self.h_par_train = h_par_train
        
        self.img_list = []
        self.G_losses = []
        self.D_losses = []
        self.fid = None
        self.iters = 0
        
        self.criterion = nn.BCELoss()
        # Batch of latent vectors to visualize the progression of generator
        # FIXME: Why 64? Const 8 x 8 plot output?
        self.fixed_noise = torch.randn(64, self.h_par_model["nz"], 1, 1, device=self.device)
        # Establish convention for real and fake labels during training
        self.real_label = 1.
        self.fake_label = 0.
        
        # Setup Adam optimizers for both G and D
        self.optimizerD = optim.Adam(
            self.netD.parameters(), lr=self.h_par_train["lr"], betas=(self.h_par_train["beta1"], 0.999)
        )
        self.optimizerG = optim.Adam(
            self.netG.parameters(), lr=self.h_par_train["lr"], betas=(self.h_par_train["beta1"], 0.999)
        )
        # optimizerD = optim.Adam(netD.parameters(), lr=lr*0.6, betas=(beta1, 0.999))
        # optimizerG = optim.Adam(netG.parameters(), lr=lr*2., betas=(beta1, 0.999))
        
    def train(self, num_epochs = None, bfid_study: bool = False, bverbose: bool = False):
        """Training Loop"""

        if num_epochs is None:
            num_epochs = self.h_par_train["num_epochs"]
    
        print("Starting Training Loop...")

        self.netD.train()
        self.netG.train()
        if bfid_study:
            fid_list = []
        bnew_fid = True
        fid_step = 10

        for epoch in range(num_epochs):
            for i, data in enumerate(self.dataloader, 0):

                if self.iters < self.h_par_train["max_iter_noise"] and self.h_par_train["bdiscnoise"]:
                    sigma = (1 - self.iters / self.h_par_train["max_iter_noise"])
                else:
                    sigma = 0.

                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                self.netD.zero_grad()
                # Format batch
                # noise = torch.randn(data[0].shape)
                real_cpu = data[0].to(self.device)
                if sigma != 0.:
                    disc_noise = torch.normal(mean=0., std=sigma, size=data[0].shape)
                    real_cpu += disc_noise.to(self.device)
                b_size = real_cpu.size(0)
                
                if self.h_par_train["bsmoothlabel"]:
                    # use smooth label for discriminator
                    label = torch.full((b_size,), (self.real_label - self.h_par_train["smoothing"]), dtype=torch.float, device=self.device)
                else:
                    label = torch.full((b_size,), self.real_label, dtype=torch.float, device=self.device)
                
                
                # Forward pass real batch through D
                output = self.netD(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = self.criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()
        
                ## Train with all-fake batch
                # Generate batch of latent vectors from gaussian noise as g input
                noise = torch.randn(b_size, self.h_par_model["nz"], 1, 1, device=self.device)
                # Generate fake image batch with G
                # noise = torch.randn(data[0].shape)
                fake = self.netG(noise)
                label.fill_(self.fake_label)
                # Classify all fake batch with D
                if sigma != 0.:
                    disc_noise = torch.normal(mean=0., std=sigma, size=data[0].shape)
                    output = self.netD((fake + disc_noise.to(self.device)).detach()).view(-1)
                else:
                    output = self.netD(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = self.criterion(output, label)
                # Calculate the gradients for this batch
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake
                # Update D
                if self.h_par_train["bthresh"] and self.iters > 10:
                    if self.D_losses[-1] >= self.h_par_train["dthres"]:
                        self.optimizerD.step()
                else:
                    self.optimizerD.step()
                    
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.netG.zero_grad()
                # fake labels are real for generator cost
                # Flip labels when training generator: real = fake, fake = real
                label.fill_(self.real_label) 
                # Since we just updated D, perform another forward pass of all-fake batch through D
                if sigma != 0.:
                    disc_noise = torch.normal(mean=0., std=sigma, size=data[0].shape)
                    output = self.netD((fake + disc_noise.to(self.device))).view(-1)
                else:
                    output = self.netD(fake).view(-1)
                # Calculate G's loss based on this output
                errG = self.criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                if self.h_par_train["bthresh"] and self.iters > 10:
                    if self.G_losses[-1] >= self.h_par_train["gthres"]:
                        self.optimizerG.step()
                else:
                    self.optimizerG.step()
        
                # Output training stats
                if bverbose:
                    if i % 100 == 0:
                        print(
                            f"[{epoch + 1}/{num_epochs}][{i + 1}/{len(self.dataloader)}]\t"
                            f"DNoise Sig: {sigma:0.3f}\t"
                            f"Loss_D: {errD.item():0.4f}\tLoss_G: {errG.item():0.4f}\t"
                            f"D(x): {D_x:0.4f}\tD(G(z)): {D_G_z1:0.4f} / {D_G_z2:0.4f}"
                        )
                            
                self.G_losses.append(errG.item())
                self.D_losses.append(errD.item())
        
                # Check how the generator is doing by saving G's output on fixed_noise
                if (self.iters % 250 == 0) or ((epoch == num_epochs-1) and (i == len(self.dataloader)-1)):
                    with torch.no_grad():
                        fake = self.netG(self.fixed_noise).detach().cpu()
                    self.img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                    self.gan_net.save_models(tag="tmp", outdir=".")
                    self.save_object([self.iters, sigma, self.G_losses, self.D_losses, self.img_list, ], "run_infos.p")  
                    
                self.iters += 1
            
            if bfid_study:
                if (epoch % fid_step == 0) and epoch > 1:
                    self.fid = self.calc_fid()
                    fid_list.append(np.asarray([epoch, self.fid]))
                    bnew_fid = True
                    if bverbose:
                        print(f"Curr FID: {self.fid}")
            
            if self.fid is not None:
                if self.fid <= 170:
                    fid_step = 5
                    if self.fid <= 140:
                        fid_step = 3
                        if self.fid <= 130:
                            fid_step = 2
                else:
                    fid_step = 10
                
                if self.fid <= 140. and bnew_fid == True:
                    print(f"Found good one FID: {self.fid:0.2f}")
                    self.gan_net.save_models(tag=f"fid{int(self.fid)}_{epoch}", outdir=".")
                    bnew_fid = False
                    
        self.fid = self.calc_fid()
        fid_list.append(np.asarray([epoch, self.fid]))
        if bverbose:
            print(f"FID after training: {self.fid:0.3f}")
        fid_list = np.asarray(fid_list)
        
        if bfid_study:
            if bverbose:
                print(f"FID list:\n {fid_list.T}")
                print(f"Best FID value: {fid_list.T[1].min()}")
                
                plt.plot(fid_list.T[0], fid_list.T[1])
                plt.show()
                
            return [fid_list.T[0][fid_list.T[1].argmin()], fid_list.T[1].min()]
        else:
            return [num_epochs, self.fid]
    
    @staticmethod
    def save_object(obj, filename):
        with open(filename, 'wb') as output:
            pickle.dump(obj, output, -1)
    
    def plot_losses(self):
        """Plot loss curves"""
        
        plt.figure(figsize=(17,10))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.G_losses,label="G")
        plt.plot(self.D_losses,label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        
    def plot_comparison(self):
        """Plot comparison for real and generated samples from last epoch"""
        
        real_batch = next(iter(self.dataloader))
        
        plt.figure(figsize=(30,30))
        plt.subplot(1,2,1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(
            np.transpose(
                vutils.make_grid(real_batch[0].to(self.device)[:64], padding=5, normalize=True).cpu(),(1,2,0)
            ) 
        )
        
        plt.subplot(1,2,2)
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(self.img_list[-1],(1,2,0)))
        plt.show()
        
    def gen_fakes(self, n_gen_images: int = 6, upscale: float = 2.):
        """"""
        
        self.netG.eval()
        
        noise = torch.randn(n_gen_images, self.h_par_model["nz"], 1, 1, device=self.device)
        fake = self.netG(noise).detach().cpu()
        
        up = torch.nn.Upsample(scale_factor=upscale)
        fake = up(fake)
        
        fig, axs = plt.subplots(1, n_gen_images, figsize=(25, 25))
        
        for ind, fake_im in enumerate(fake):
            """
            im = transforms.ToPILImage()(fake_im[0]).convert("RGB")
            enh1 = ImageEnhance.Contrast(im)
            fac1 = 1.1
            enh2 = ImageEnhance.Sharpness(im)
            fac2 = 1.1
            im_output = enh1.enhance(fac1)
            im_output = enh2.enhance(fac2)
            
            im = transforms.ToTensor()(im_output)
            
            """
            im = vutils.make_grid(fake_im, padding=2, normalize=True)
            im = np.transpose(im, (1, 2, 0))
            axs.flat[ind].imshow(im)
            axs.flat[ind].axis("off")
        plt.show()

    def _create_tmp_images(self, path: str, no_images: int):
        """Create sample fake images for FID calculation"""
        
        self.netG.eval()
        for ind in range(no_images):
            noise = torch.randn(1, self.h_par_model["nz"], 1, 1, device=self.device)
            fake = self.netG(noise).detach().cpu()[0]
            
            save_image(fake, path + f"/{ind}.jpg")

    def _create_tmp_reals(self, path: str, no_images: int):
        """Copy sample real images for FID calculation"""
        
        real_batch = next(iter(self.dataloader))
        for ind in range(no_images):
            real = real_batch[0].to(self.device)[ind]
            save_image(real, path + f"/{ind}.jpg")
        
    def calc_fid(self, no_images: int = 128):
        """Use pytorch_fid to calculate fid value for real and generated images"""
        
        real_path = "/content/gdrive/MyDrive/FrogGAN/tmp/tmp_reals"
        fake_path = "/content/gdrive/MyDrive/FrogGAN/tmp/tmp_fakes"
        self._create_tmp_images(path=fake_path, no_images=no_images)
        self._create_tmp_reals(path=real_path, no_images=no_images)
        
        return calculate_fid_given_paths([real_path, fake_path], batch_size=no_images, device=self.device, dims=2048)