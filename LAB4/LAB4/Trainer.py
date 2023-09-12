import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10

def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= batch_size  
  return KLD


class kl_annealing():
    def __init__(self, args, current_epoch=0):
        self.strategy = args.strategy
        self.beta = 0.0
        self.max_beta = 1.0
        self.anneal_rate = args.anneal_rate
        self.anneal_interval = args.anneal_interval
        self.current_epoch = current_epoch
        self.cycle_length = args.cycle_length if self.strategy == 'cyclical' else None
        
    def update(self):
        if self.strategy == 'cyclical':
            cycle_position = self.current_epoch % self.cycle_length
            self.beta = min(self.max_beta, cycle_position / (self.cycle_length / 2))
            if cycle_position > self.cycle_length / 2:
                self.beta = self.max_beta - (cycle_position - self.cycle_length / 2) / (self.cycle_length / 2) * self.max_beta
                linear_schedule = self.frange_cycle_linear(self.cycle_length, start=0.0, stop=self.max_beta)
                self.beta = linear_schedule[cycle_position]  

        elif self.strategy == 'monotonic':
            if self.current_epoch % self.anneal_interval == 0:
                self.beta += self.anneal_rate
                self.beta = min(self.beta, self.max_beta)
                
        elif self.strategy == 'none':
            self.beta = self.max_beta
        
        self.current_epoch += 1
        
    def get_beta(self):
        return self.beta
        
    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0, n_cycle=4, ratio=0.5):
        L = np.ones(n_iter) * stop
        period = n_iter / n_cycle
        step = (stop - start) / (period * ratio)  # linear schedule

        for c in range(n_cycle):
            v, i = start, 0
            while v <= stop and (int(i + c * period) < n_iter):
                L[int(i + c * period)] = v
                v += step
                i += 1
        return L
        

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 5], gamma=0.1)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 1
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size
        self.best_psnr = 0
        self.epoch_counter = 1
        self.last_generated_frame = None
        self.last_img_frame = None
        self.last_label_frame = None
        self.losses = []
        self.psnrs = []
        self.teacher_forcing_ratios = []
        
        
    def forward(self, img, label):
        pass
    
    def training_stage(self):
        for i in range(self.args.num_epoch):
            lss = []
            train_loader = self.train_dataloader()
            
            for (img, label) in (pbar := tqdm(train_loader, ncols=120)):
                adapt_TeacherForcing = True if random.random() < self.tfr else False
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss = self.training_one_step(img, label, adapt_TeacherForcing)
                lss.append(loss.item())
                
                beta = self.kl_annealing.get_beta()
                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.2f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.2f}], beta: {}'.format(self.tfr, beta), pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
                loss.backward()
                self.optimizer_step()
                self.scheduler.step()
            
            self.losses.append(sum(lss)/(len(lss)))
            self.teacher_forcing_ratios.append(self.tfr)
            self.eval()
            self.current_epoch += 1
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()
            self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch-1}.ckpt"))
        self.plot_curves()
            
            
            
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        for img, label in val_loader:
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            psnr = self.val_one_step(img, label)
        print(f'epoch:{self.current_epoch} ,PSNR:{psnr}')
        print()
        if psnr > self.best_psnr:
            self.best_psnr = psnr
            self.save(os.path.join(self.args.save_root, "best_model.ckpt"))
            
    
    def training_one_step(self, img, label, adapt_TeacherForcing):
        img = img.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
        label = label.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
        decoded_frame_list = [img[0].cpu()]
        human_feat_hat_list = []
        label_list = []
        seq_loss = 0
        for i in range(0, self.train_vi_len):
            human_feat_hat = self.frame_transformation(img[i])
            human_feat_hat_list.append(human_feat_hat)
            
        for i in range(1, self.train_vi_len):
            label_feat = self.label_transformation(label[i])
            z, mu, log_var = self.Gaussian_Predictor(human_feat_hat_list[i], label_feat)
            if not adapt_TeacherForcing:
                human_feat_hat = self.frame_transformation(self.last_generated_frame)
            else:
                human_feat_hat = human_feat_hat_list[i-1]
            parm = self.Decoder_Fusion(human_feat_hat, label_feat, z) 
            generated_frame = self.Generator(parm)
            reconstruction_loss = self.mse_criterion(generated_frame, img[i])
            self.last_generated_frame = generated_frame.detach()
            beta = self.kl_annealing.get_beta()
            loss = reconstruction_loss  + beta * kl_criterion(mu, log_var, self.batch_size)
            seq_loss = seq_loss + loss
            
        return seq_loss / self.train_vi_len
            
        
    def val_one_step(self, img, label, idx=0):
        img = img.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
        label = label.permute(1, 0, 2, 3, 4) # change tensor into (seq, B, C, H, W)
        
        decoded_frame_list = [img[0].cpu()]
        label_list = []

        # Normal normal
        out = img[0]
        
        for i in range(1, self.val_vi_len):
            z = torch.cuda.FloatTensor(1, self.args.N_dim, self.args.frame_H, self.args.frame_W).normal_()
            label_feat = self.label_transformation(label[i])
            human_feat_hat = self.frame_transformation(out)
            #z, _, _ = self.Gaussian_Predictor(human_feat_hat, label_feat)
            
            parm = self.Decoder_Fusion(human_feat_hat, label_feat, z)    
            out = self.Generator(parm)
            
            decoded_frame_list.append(out.cpu())
            label_list.append(label[i].cpu())
            
        
        generated_frame = stack(decoded_frame_list).permute(1, 0, 2, 3, 4)
        img = img.permute(1, 0, 2, 3, 4)
        gen_image = generated_frame[0]
        ground_truth = img[0]
        ground_truth = ground_truth.to(self.args.device)
        gen_image = gen_image.to(self.args.device)
        PSNR_LIST = []
        for i in range(1, 630):
            PSNR = Generate_PSNR(ground_truth[i], gen_image[i])
            PSNR_LIST.append(PSNR.item())
        return sum(PSNR_LIST)/(len(PSNR_LIST)-1)
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        if self.current_epoch % 5 == 0:
            self.tfr = max(self.tfr - 0.05, 0)
            
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "losses"    : self.losses,
            "tfr"       :   self.tfr,
            "tft_length" : self.teacher_forcing_ratios,
            "last_epoch": self.current_epoch,
            "last_gen_frame" : self.last_generated_frame
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            self.losses = checkpoint['losses']
            self.last_generated_frame = checkpoint['last_gen_frame']
            self.teacher_forcing_ratios = checkpoint["tft_length"]
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=1)
            self.current_epoch = checkpoint['last_epoch'] + 1
            self.kl_annealing = kl_annealing(self.args, self.current_epoch)

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()
        
    def plot_curves(self):
    # Plotting loss curve
        plt.figure()
        plt.plot(range(1, len(self.losses) + 1), self.losses)
        plt.title('KL annealing (Cyclical)_Loss Curve')
        plt.xlabel('epochs')
        plt.ylabel('Loss')
        #plt.xticks(list(range(1, len(self.losses) + 1)))
        plt.savefig('loss_curve.png')

        #plt.figure()
        #plt.plot(self.psnrs)
        #plt.title('PSNR Curve')
        #plt.xlabel('Iterations')
        #plt.ylabel('PSNR')
        #plt.savefig('psnr_curve.png')

    # Plotting Teacher Forcing Ratio curve
        plt.figure()
        plt.plot(range(1, len(self.teacher_forcing_ratios) + 1), self.teacher_forcing_ratios)
        plt.title('Teacher Forcing Ratio Curve')
        plt.xlabel('epochs')
        plt.ylabel('Teacher Forcing Ratio')
        #plt.xticks(list(range(1, len(self.teacher_forcing_ratios) + 1)))
        plt.savefig('teacher_forcing_ratio_curve.png')



def main(args):
    
    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, default='C:/Users/user/Desktop/LAB4_dataset/LAB4_Dataset',  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, default='C:/Users/user/Desktop/weight',  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=1,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default="C:/Users/user/Desktop/epoch=80.ckpt" ,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--strategy', type=str, default='cyclical', choices=['cyclical', 'monotonic', 'none'], 
                    help='Type of KL annealing strategy to use.')
    parser.add_argument('--anneal_rate', type=float, default=0.01, 
                    help='Rate at which to increase the beta value for monotonic strategy.')
    parser.add_argument('--anneal_interval', type=int, default=1, 
                    help='Interval (in epochs) at which to update the beta value for monotonic strategy.')
    parser.add_argument('--cycle_length', type=int, default=10, 
                    help='Number of epochs for a complete cycle (increase and decrease) for cyclical strategy.')
    

    

    args = parser.parse_args()
    
    main(args)
