"""
    The base code is taken from public GitHup repository of Zhongyang-debug, and new methods proposed
    in the paper are implemented by

        - Süleyman Ateş - ates.suleyman@metu.edu.tr
        - Arda Numanoğlu - arda.numanoglu@metu.edu.tr
        
    GitHub repository of Zhongyang-debug including reproduction of SepFormer model, can be found at:
    https://github.com/Zhongyang-debug/Attention-Is-All-You-Need-In-Speech-Separation

"""

import os
import time
from sklearn.utils import resample
import torch
from src.pit_criterion import cal_loss_pit, cal_loss_no, MixerMSE
from torch.utils.tensorboard import SummaryWriter
import gc
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T


class Trainer(object):
    def __init__(self, data, model ,optimizer, config):

        self.tr_loader = data["tr_loader"]
        self.cv_loader = data["cv_loader"]
        self.model = model
        self.optimizer = optimizer

        # Training config
        self.use_cuda = config["train"]["use_cuda"]
        self.epochs = config["train"]["epochs"]
        self.half_lr = config["train"]["half_lr"]
        self.early_stop = config["train"]["early_stop"]
        self.max_norm = config["train"]["max_norm"]

        # save and load model
        self.save_folder = config["save_load"]["save_folder"]
        self.checkpoint = config["save_load"]["checkpoint"]
        self.continue_from = config["save_load"]["continue_from"]
        self.model_path = config["save_load"]["model_path"]

        # logging
        self.print_freq = config["logging"]["print_freq"]

        # loss
        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)

        os.makedirs(self.save_folder, exist_ok=True)
        self.prev_val_loss = float("inf")
        self.best_val_loss = float("inf")
        self.halving = False
        self.val_no_improve = 0

        self.write = SummaryWriter("./logs")

        self._reset()

        self.MixerMSE = MixerMSE()

        self.loss_history = []

    def _reset(self):
        if self.continue_from:
            
            print('Loading checkpoint model %s' % self.continue_from)
            package = torch.load(self.save_folder + self.continue_from)

            if isinstance(self.model, torch.nn.DataParallel):
                self.model = self.model.module

            self.model.load_state_dict(package['state_dict'])
            self.optimizer.load_state_dict(package['optim_dict'])

            self.start_epoch = int(package.get('epoch', 1))

            self.tr_loss[:self.start_epoch] = package['tr_loss'][:self.start_epoch]
            self.cv_loss[:self.start_epoch] = package['cv_loss'][:self.start_epoch]
        else:
            
            self.start_epoch = 0

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            print("Train Start...")

            self.model.train()

            start_time = time.time()

            tr_loss = self._run_one_epoch(epoch)

            gc.collect()
            torch.cuda.empty_cache()

            self.write.add_scalar("train loss", tr_loss, epoch+1)

            end_time = time.time()
            run_time = end_time - start_time

            print('-' * 85)
            print('End of Epoch {0} | Time {1:.2f}s | Train Loss {2:.3f}'.format(epoch+1, run_time, tr_loss))
            print('-' * 85)

            if self.checkpoint:
                
                file_path = os.path.join(self.save_folder, 'epoch%d.pth.tar' % (epoch + 1))

                if self.continue_from == "":
                    if isinstance(self.model, torch.nn.DataParallel):
                        self.model = self.model.module

                torch.save(self.model.serialize(self.model,
                                                self.optimizer,
                                                epoch + 1,
                                                tr_loss=self.tr_loss,
                                                cv_loss=self.cv_loss), file_path)

                print('Saving checkpoint model to %s' % file_path)

            print('Cross validation Start...')

            self.model.eval()

            start_time = time.time()

            val_loss = self._run_one_epoch(epoch, cross_valid=True)

            self.write.add_scalar("validation loss", val_loss, epoch+1)

            end_time = time.time()
            run_time = end_time - start_time

            print('-' * 85)
            print('End of Epoch {0} | Time {1:.2f}s | ''Valid Loss {2:.3f}'.format(epoch+1, run_time, val_loss))
            print('-' * 85)

            if self.half_lr:
                
                if val_loss >= self.prev_val_loss:
                    self.val_no_improve += 1

                    if self.val_no_improve >= 3:
                        self.halving = True

                    if self.val_no_improve >= 10 and self.early_stop:
                        print("No improvement for 10 epochs, early stopping.")
                        break
                else:
                    self.val_no_improve = 0

            if self.halving:
                optime_state = self.optimizer.state_dict()
                optime_state['param_groups'][0]['lr'] = optime_state['param_groups'][0]['lr']/2.0
                self.optimizer.load_state_dict(optime_state)
                print('Learning rate adjusted to: {lr:.6f}'.format(lr=optime_state['param_groups'][0]['lr']))
                self.halving = False

            self.prev_val_loss = val_loss

            self.tr_loss[epoch] = tr_loss
            self.cv_loss[epoch] = val_loss

            if val_loss < self.best_val_loss:

                self.best_val_loss = val_loss

                file_path = os.path.join(self.save_folder, self.model_path)

                torch.save(self.model.serialize(self.model,
                                                self.optimizer,
                                                epoch + 1,
                                                tr_loss=self.tr_loss,
                                                cv_loss=self.cv_loss), file_path)

                print("Find better validated model, saving to %s" % file_path)
        return self.loss_history

    def _run_one_epoch(self, epoch, cross_valid=False):

        start_time = time.time()

        total_loss = 0
        data_loader = self.tr_loader if not cross_valid else self.cv_loader

        for i, (data) in enumerate(data_loader):

            padded_mixture, mixture_lengths, padded_source = data

            if torch.cuda.is_available():
                padded_mixture = padded_mixture.cuda()
                mixture_lengths = mixture_lengths.cuda()
                padded_source = padded_source.cuda()
            
            # MULTI-LOSS IMPLEMENTATION HERE
            resample_array = [500, 1000, 2000, 3000, 4000, 5000]
            target_dsample_array = list()
            for dsample in resample_array:
                resample_rate = dsample
                resampler = T.Resample(8000, resample_rate, dtype=padded_source.dtype)
                if torch.cuda.is_available():
                  resampler = resampler.cuda()
                target_dsample_array.append(resampler(padded_source))
            
            # 7. and 8. sources are not resampled
            target_dsample_array.append(padded_source) # 7.
            target_dsample_array.append(padded_source) # 8.
            
            sep_loss = 0
            sep_out_list, superres_out = self.model(padded_mixture)
            for k, output in enumerate(sep_out_list):
              if k < 6:
                resampler = T.Resample(8000, resample_array[k], dtype=padded_source.dtype)
                if torch.cuda.is_available():
                  resampler = resampler.cuda()
                output = resampler(output)
                
              loss_temp, max_snr, estimate_source, reorder_estimate_source = cal_loss_pit(target_dsample_array[k],
                                                                                      output,
                                                                                      mixture_lengths)
              sep_loss += loss_temp

            sep_loss /= 8

            sres_loss, max_snr, estimate_source, reorder_estimate_source = cal_loss_pit(padded_source,
                                                                                      superres_out,
                                                                                      mixture_lengths)
            
            final_loss = sep_loss + sres_loss

            if not cross_valid:
                self.optimizer.zero_grad()
                final_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                self.optimizer.step()

            if not cross_valid:
                self.loss_history.append(final_loss.item())

            total_loss += final_loss.item()

            end_time = time.time()
            run_time = end_time - start_time

            if i % self.print_freq == 0:
                print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | Current Loss {3:.6f} | {4:.1f} s/batch'.format(
                    epoch+1,
                    i+1,
                    total_loss/(i+1),
                    final_loss.item(),
                    run_time/(i+1)),
                    flush=True)

        return total_loss/(i+1)
