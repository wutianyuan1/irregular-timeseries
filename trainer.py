import torch
import os
import sys
import yaml
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np

from datetime import datetime
from tqdm.auto import tqdm
from dataset import Walker2dImitationData, WalkerTorchDataset
from torch.utils.tensorboard import SummaryWriter
from models import S4Model, BaselineModel


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Trainer(object):
    def __init__(self, config, trainset, valset, testset) -> None:
        self.config = config
        self.d_input = trainset.x.shape[-1]
        self.d_output = self.d_input
        self.trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.config['batch_size'], shuffle=True, num_workers=self.config['num_workers'])
        self.valloader = torch.utils.data.DataLoader(
            valset, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.config['num_workers'])
        self.testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.config['num_workers'])
        if device == 'cuda':
            cudnn.benchmark = True
        self.model = self.build_model(self.config['model'], self.d_input, self.d_output)
        self.start_time = f"{datetime.now()}".replace(" ", '-')
        self.logdir = 'log/sbexpr-{}-{}'.format(self.config['model'], self.start_time)
        self.writer = SummaryWriter(self.logdir)
        self.optimizer, self.scheduler = self.setup_optimizer(
            self.model, self.config['lr'], self.config['weight_decay'], self.config['epochs'])
        self.criterion = nn.MSELoss()
        if os.path.isdir(self.config['resume_path']):
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            checkpoint = torch.load(self.config['resume_path'])
            self.model.load_state_dict(checkpoint['model'])
        # log steps and best values
        self.train_log_step = 0
        self.eval_log_step  = 0
        self.save_log_step  = 0
        self.best_loss      = float('inf')  # best eval MSE loss


    def build_model(self, model_name, d_input, d_output):
        # Model
        print('==> Building model..')
        if model_name == 's4':
            model = S4Model(
                d_input=d_input,
                d_output=d_output,
                d_model=self.config['d_model'],
                n_layers=self.config['n_layers'],
                dropout=self.config['dropout'],
                prenorm=self.config['prenorm'],
                lr=self.config['lr'],
            )
        else:
            model = BaselineModel(
                model_name=model_name,
                d_input=d_input,
                d_output=d_output,
                n_layers=self.config['n_layers'],
            )
        model = model.to(device)
        return model

    def setup_optimizer(self, model, lr, weight_decay, epochs):
        """
        S4 requires a specific optimizer setup.

        The S4 layer (A, B, C, dt) parameters typically
        require a smaller learning rate (typically 0.001), with no weight decay.

        The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
        and weight decay (if desired).
        """
        # All parameters in the model
        all_parameters = list(model.parameters())

        # General parameters don't contain the special _optim key
        params = [p for p in all_parameters if not hasattr(p, "_optim")]

        # Create an optimizer with the general parameters
        optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

        # Add parameters with special hyperparameters
        hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
        hps = [
            dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
        ]  # Unique dicts
        for hp in hps:
            params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
            optimizer.add_param_group(
                {"params": params, **hp}
            )

        # Create a lr scheduler
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        # Print optimizer info
        keys = sorted(set([k for hp in hps for k in hp.keys()]))
        for i, g in enumerate(optimizer.param_groups):
            group_hps = {k: g.get(k, None) for k in keys}
            print(' | '.join([
                f"Optimizer group {i}",
                f"{len(g['params'])} tensors",
            ] + [f"{k} {v}" for k, v in group_hps.items()]))

        return optimizer, scheduler

    # Training
    def train_epoch(self):
        self.model.train()
        train_loss = 0
        total = 0
        pbar = tqdm(enumerate(self.trainloader))
        for batch_idx, (inputs, targets, times) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs, times)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            total += targets.size(0)
            self.writer.add_scalar("train_loss", loss.item(), self.train_log_step)
            self.train_log_step += 1
            pbar.set_description(
                'Batch Idx: (%d/%d) | Loss: %.3f | MSE: %.3f' %
                (batch_idx, len(self.trainloader), train_loss/(batch_idx+1), loss.item())
            )

    def eval_epoch(self, epoch, dataloader, checkpoint=False):
        self.model.eval()
        eval_loss = 0
        with torch.no_grad():
            pbar = tqdm(enumerate(dataloader))
            for batch_idx, (inputs, targets, times) in pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs, times)
                loss = self.criterion(outputs, targets)
                eval_loss += loss.item()
                self.writer.add_scalar("eval_loss", loss.item(), self.eval_log_step)
                self.eval_log_step += 1
                pbar.set_description(
                    'Batch Idx: (%d/%d) | Loss: %.3f | MSE: %.3f' %
                    (batch_idx, len(dataloader), eval_loss/(batch_idx+1), loss.item())
                )

        # Save checkpoint.
        curr_loss = eval_loss/(batch_idx+1)
        if checkpoint and curr_loss < self.best_loss:
            state = {
                'model': self.model.state_dict(),
                'loss': curr_loss,
                'epoch': epoch,
            }
            torch.save(state, self.logdir + '/ckpt.pt')
            self.writer.add_scalar("best_loss", curr_loss, self.save_log_step)
            self.save_log_step += 1
            self.best_loss = curr_loss

        return curr_loss

    def run(self):
        print("total parameters:", self.model.param_count())
        self.writer.add_scalar("param_count", self.model.param_count(), 0)
        pbar = tqdm(range(0, self.config['epochs']))
        val_curr_loss = float("inf")
        for epoch in pbar:
            if epoch == 0:
                pbar.set_description('Epoch: %d' % (epoch))
            else:
                pbar.set_description('Epoch: %d | Val curr_loss: %1.3f' % (epoch, val_curr_loss))
            self.train_epoch()
            val_curr_loss = self.eval_epoch(epoch, self.valloader, checkpoint=True)
            self.eval_epoch(epoch, self.testloader)
            self.scheduler.step()
    
    def print_seq(input, predict, target):
        f = open("predict.csv", 'a+')
        l = torch.nn.MSELoss()
        for i in range(len(predict)):
            f.write("predict={}\n, target={}\n, loss={}\n\n\n".format(
                                            list(predict[i].cpu().detach().numpy()),
                                            list(target[i].cpu().detach().numpy()),
                                            l(predict[i], target[i]) ))
        f.close()


if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f:
        config = yaml.safe_load(f)
    if len(sys.argv) > 2:
        config['model'] = sys.argv[2]
    allset = Walker2dImitationData(seq_len=64)
    trainset = WalkerTorchDataset(allset.train_x, allset.train_y, allset.train_times)
    valset = WalkerTorchDataset(allset.valid_x, allset.valid_y, allset.valid_times)
    testset = WalkerTorchDataset(allset.test_x, allset.test_y, allset.test_times)
    trainer = Trainer(config, trainset, valset, testset)
    trainer.run()
