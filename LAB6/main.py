import os
import random
import numpy as np
import argparse
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMScheduler
from model import newUNet2D
from dataloader import ICLEVRDataset
from utils import evaluate
from evaluator import evaluation_model

torch.cuda.empty_cache()


class DDPM(nn.Module):
    def __init__(self, args):
        super(DDPM, self).__init__()
        self.args = args
        self.sample_size = args.sample_size
        self.block_dim = args.block_dim
        self.layers = args.layers_per_block
        self.embed_type = args.embed_type
        self.num_epochs = args.num_epochs
        self.lr = args.lr
        self.batch_size = args.batch_size

        self.model = newUNet2D(
            sample_size=args.sample_size,      
            in_channels=3,                      
            out_channels=3,
            layers_per_block=args.layers_per_block,
            block_out_channels=(args.block_dim, args.block_dim, args.block_dim*2, args.block_dim*2, args.block_dim*4, args.block_dim*4),
            down_block_types=(
                "DownBlock2D",          
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",      
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",        
                "UpBlock2D",           
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
        self.beta_schedule = "squaredcos_cap_v2" if args.beta_schedule == "cosine" else "linear"
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000, prediction_type=args.predict_type, beta_schedule=self.beta_schedule)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.train_loader = self.train_dataloader()
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=len(self.train_loader) * self.num_epochs,
        )
        self.evaluation = evaluation_model() 
        self.losses = []
        self.acc_list = []
        self.new_acc_list = []
        self.best_acc = 0
        self.best_new_acc = 0
        self.global_step = 0
        self.test_acc = 0
        self.new_test_acc = 0
        self.current_epoch = 0

        self.accelerator = Accelerator(
            mixed_precision=args.mixed_precision,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir=os.path.join(args.log_dir, "logging"),
        )

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers("train_example")

        self.model, self.optimizer, self.train_loader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.lr_scheduler
        )
        
    def train_dataloader(self):        
        transform = transforms.Compose([
            transforms.Resize((self.sample_size, self.sample_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        dataset = ICLEVRDataset(args, mode='train', transforms=transform)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=8, pin_memory=True, shuffle=True)
        
        return train_loader

    def training_stage(self):
        for epoch in range(self.num_epochs):
            epoch = self.current_epoch
            progress_bar = tqdm(total=len(self.train_loader), disable=not self.accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch+1}/{self.num_epochs}")
            total_loss = 0
            for i, (x, class_label) in enumerate(self.train_loader):
                x, class_label = x.to(self.args.device), class_label.to(self.args.device)     
                noise = torch.randn_like(x)
                timesteps = torch.randint(0, 1000, (x.shape[0],)).long().to(self.args.device)
                noisy_image = self.noise_scheduler.add_noise(x, noise, timesteps)
                with self.accelerator.accumulate(self.model):
                    noise_pred = self.model(noisy_image, timesteps, class_label).sample
                    loss = self.loss_fn(noise_pred, noise)
                    total_loss += loss.item()
                    self.accelerator.backward(loss)
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                logs = {"loss": total_loss / (i+1), "lr": self.lr_scheduler.get_last_lr()[0], "step": self.global_step}
                progress_bar.update(1)
                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=self.global_step)
                self.global_step += 1
            self.losses.append(total_loss / len(self.train_loader))
            if epoch % 5 == 0 or epoch == self.num_epochs - 1:
                self.classification_stage(epoch)
            self.current_epoch += 1
                                   
            
    def classification_stage(self, epoch):
        test_image, test_label = evaluate(self.model, self.noise_scheduler, epoch, args, self.args.device, "test")
        new_test_image, new_test_label = evaluate(self.model, self.noise_scheduler, epoch, args, self.args.device, "new_test")
        test_acc = self.evaluation.eval(test_image, test_label)
        new_test_acc = self.evaluation.eval(new_test_image, new_test_label)
        self.acc_list.append(test_acc)
        self.new_acc_list.append(new_test_acc)
        print()
        print("> Accuracy: [Test]: {:.4f}, [New Test]: {:.4f}".format(test_acc, new_test_acc))

        with open('{}/train_record.txt'.format(self.args.log_dir), 'a') as train_record:
            train_record.write(('[Epoch: %02d] loss: %.5f | test acc: %.5f | new_test acc: %.5f\n' % (epoch, self.losses[-1], test_acc, new_test_acc)))

        if test_acc >= self.best_acc and new_test_acc >= self.best_new_acc:
            self.best_acc = test_acc
            self.best_new_acc = new_test_acc
            self.save(os.path.join(self.args.log_dir, "model.pth"))
            print("save best weight")
            
    def evals(self):
        test_image, test_label = evaluate(self.model, self.noise_scheduler, 150, args, self.args.device, "test")
        new_test_image, new_test_label = evaluate(self.model, self.noise_scheduler, 150, args, self.args.device, "new_test")
        test_acc = self.evaluation.eval(test_image, test_label)
        new_test_acc = self.evaluation.eval(new_test_image, new_test_label)
        print()
        print("> Accuracy: [Test]: {:.4f}, [New Test]: {:.4f}".format(test_acc, new_test_acc))

            
    def save(self, path):
        torch.save({
            "model": self.model.state_dict(),
            "acc" : self.acc_list,
            "new_acc" : self.new_acc_list,
            "test_acc":self.best_acc,
            "new_test_acc":self.best_new_acc,
            "last_epoch": self.current_epoch
        }, path)
        
    def load_weight(self):
        state = torch.load("model.pth")
        self.model.load_state_dict(state['model'], strict=True) 




            
def main(args):
    
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    name = 'lr={:.5f}-lr_warmup={}-block_dim={}-layers={}-schedule={}-predict_type={}'.format(args.lr, args.lr_warmup_steps, args.block_dim, args.layers_per_block, args.beta_schedule, args.predict_type)
    args.log_dir = './%s/%s' % (args.log_dir, name)
    args.figure_dir = '%s/%s' % (args.log_dir, args.figure_dir)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.figure_dir, exist_ok=True)
    with open('{}/train_record.txt'.format(args.log_dir), 'w') as train_record:
        train_record.write('args: {}\n'.format(args))
    
    models = DDPM(args).to(args.device)    
    if args.test:
        models.load_weight()
        models.evals()
    else:
        models.training_stage()
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default="./dataset", type=str)
    parser.add_argument('--log_dir', default="logs", type=str)
    parser.add_argument('--figure_dir', default="figures", type=str)
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--num_epochs', default=120, type=int) 
    parser.add_argument('--sample_size', default=64, type=int)
    parser.add_argument('--beta_schedule', default="cosine", type=str)
    parser.add_argument('--predict_type', default="epsilon", type=str)
    parser.add_argument('--block_dim', default=128, type=int)
    parser.add_argument('--layers_per_block', default=2, type=int)
    parser.add_argument('--embed_type', default="timestep", type=str)
    parser.add_argument('--lr_warmup_steps', default=500, type=int)
    parser.add_argument('--mixed_precision', default="fp16", type=str)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--seed', default=1, type=int)
    
    args = parser.parse_args()
    
    main(args)