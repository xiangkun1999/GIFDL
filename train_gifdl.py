


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import argparse
import random
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import time
import cv2
import scipy.io as sio
import logging
from pathlib import Path
from torch.optim.lr_scheduler import StepLR

from ResUNet1NoSkip import UNet
from xunet import XuNet
from yednet import YedNet

TS_PARA = 60
PAYLOAD = 0.4

mylambda = 4

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

def myParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='bows|bossbase|szu',default='szu')
    #cover
    parser.add_argument('--dataroot', help='path to dataset',default='/data-x/g12/zhangjiansong/stable_diffusion_G/gen1000_7.4980')
    #fluc
    parser.add_argument('--flucroot', help='path to fluction dataset',default='/data-x/g12/zhangjiansong/stable_diffusion_G/gen1000_7.')

    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=512, help='the height / width of the input image to network')

    parser.add_argument('--niter', type=int, default=72, help='number of epochs to train for')


    parser.add_argument('--lrG', type=float, default=0.00001, help='learning rate, default=0.0002')
    parser.add_argument('--lrD', type=float, default=0.00001, help='learning rate, default=0.0002')

    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netXu', default='', help="path to netXu (to continue training)")
    parser.add_argument('--netYed', default='', help="path to netYed (to continue training)")
    parser.add_argument('--outf', default='FCoL_dtof', help='folder to output images and model checkpoints')
    
    parser.add_argument('--config', default='', help='config for training')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    
    
    opt = parser.parse_args()
    return opt



    

#两个数据集合一
class SZUDataset256(data.Dataset):
    def __init__(self, root, data_dir, transforms = None):
        self.imgs = os.listdir(root)
       
        self.errordir = ['4970/', '4991/', '4960/', '4997/','4999/','5000/','5001/','5030/','5010/','5020/']
        self.data_dir = data_dir
        self.root = root
        self.transforms = transforms

    def __getitem__(self,index):
        img_path = os.path.join(self.root,self.imgs[index])
        label = np.array([0, 1], dtype='int32')

        data = cv2.imread(img_path,0)
        rand_ud = np.random.rand(512,512)
        sample = {'data':data, 'rand_ud':rand_ud, 'label':label}
        if self.transforms:
            sample = self.transforms(sample)


        myopt = random.randint(0,9)
        image_path = os.path.join(self.data_dir + self.errordir[myopt],self.imgs[index])
        data_flu = cv2.imread(image_path,0)
        sample_flu = {'data':data_flu, 'rand_ud':rand_ud, 'label':label}
        if self.transforms:
            sample_flu = self.transforms(sample_flu)
        
            
        return (sample,sample_flu)

    def __len__(self):
        return len(self.imgs)





class ToTensor():
  def __call__(self, sample):
    data,rand_ud, label = sample['data'], sample['rand_ud'], sample['label']

    data = np.expand_dims(data, axis=0)
    data = data.astype(np.float32)
    rand_ud = rand_ud.astype(np.float32)
    rand_ud = np.expand_dims(rand_ud,axis = 0)
    
    # data = data / 255.0

    new_sample = {
      'data': torch.from_numpy(data),
      'rand_ud': torch.from_numpy(rand_ud),
      'label': torch.from_numpy(label).long(),
    }

    return new_sample





class ToTensor256():
    def __call__(self, sample):
        dct, nzac, hill_cp, rand_ud, label = sample['dct'], sample['nzac'], sample['hill_cp'], sample['rand_ud'], sample['label']



        dct = np.expand_dims(dct, axis=0)
        dct = dct.astype(np.float32)

        nzac = nzac.astype(np.float32)

        hill_cp = np.expand_dims(hill_cp, axis=0)
        hill_cp = hill_cp.astype(np.float32)

        rand_ud = rand_ud.astype(np.float32)
        rand_ud = np.expand_dims(rand_ud,axis = 0)

        # prob = np.expand_dims(prob, axis=0)
        # prob = prob.astype(np.float32)



        # data = data / 255.0

        new_sample = {
            'dct': torch.from_numpy(dct),
            'nzac': torch.from_numpy(nzac),
            'hill_cp': torch.from_numpy(hill_cp),
            # 'prob': torch.from_numpy(prob),
            'rand_ud': torch.from_numpy(rand_ud),
            'label': torch.from_numpy(label).long(),

        }

        return new_sample


# custom weights initialization called on netG and netD
def weights_init_g(net):
    for m in net.modules():
        if isinstance(m,nn.Conv2d) and m.weight.requires_grad:
            m.weight.data.normal_(0., 0.02)
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0., 0.02)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0., 0.02)
            m.bias.data.fill_(0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

def weights_init_d(net):
    for m in net.modules():
        if isinstance(m,nn.Conv2d) and m.weight.requires_grad:
            m.weight.data.normal_(0., 0.01)
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0., 0.01)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0., 0.02)
            m.bias.data.fill_(0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


def weights_init_xunet(module):
  if type(module) == nn.Conv2d:
    if module.weight.requires_grad:
      nn.init.normal_(module.weight.data, mean=0, std=0.01)

      # nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')
      # nn.init.xavier_uniform_(module.weight.data)
      # nn.init.constant_(module.bias.data, val=0.2)
    # else:
    #   module.weight.requires_grad = True

  if type(module) == nn.Linear:
    nn.init.xavier_uniform_(module.weight.data)
    # nn.init.normal_(module.weight.data, mean=0, std=0.01)
    nn.init.constant_(module.bias.data, val=0)


def setLogger(log_path, mode='a'):
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)

  if not logger.handlers:
    # Logging to a file
    file_handler = logging.FileHandler(log_path, mode=mode)
    file_handler.setFormatter(logging.Formatter('%(asctime)s: %(message)s', '%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)


class AverageMeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def loss_plot(hist, path = 'Train_hist.png', model_name = ''):
    x = range(len(hist['D_loss_Xu']))

    y1 = hist['D_loss_Xu']
    y2 = hist['G_loss']
    y3 = hist['D_loss_Yed']

    
    plt.plot(x, y1, label='D_loss_Xu')
    plt.plot(x, y2, label='G_loss')
    plt.plot(x, y3, label='D_loss_Yed')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, model_name + '_loss.png')

    plt.savefig(path)

    plt.close()



def main():
    

    

    opt = myParseArgs()
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

   

    cudnn.benchmark = True
    LOG_PATH = os.path.join(opt.outf, 'model_log_'+opt.config)
    setLogger(LOG_PATH, mode = 'w')

    #device = torch.device('cuda:' + opt.gpuNum)
    
    transform = transforms.Compose([ToTensor(),])
    dataset = SZUDataset256(opt.dataroot, opt.flucroot, transforms=transform)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                            shuffle=True, num_workers=int(opt.workers),drop_last = True)


    
    netG = UNet()#.cuda()
    netG = nn.DataParallel(netG)
    netG = netG.cuda()
    netG.apply(weights_init_g)

    if opt.netG != '':
        logging.info('-' * 8)
        logging.info('Load state_dict in {}'.format(opt.netG))
        logging.info('-' * 8)
        netG.load_state_dict(torch.load(opt.netG))
    # print(netG)

    
    netXu = XuNet()#.cuda()
    netXu = nn.DataParallel(netXu)
    netXu = netXu.cuda()
    netXu.apply(weights_init_xunet)

    if opt.netXu != '':
        logging.info('-' * 8)
        logging.info('Load state_dict in {}'.format(opt.netXu))
        logging.info('-' * 8)
        netXu.load_state_dict(torch.load(opt.netXu))
    # print(netXu)

    
    netYed = YedNet()#.cuda()
    netYed = nn.DataParallel(netYed)
    netYed = netYed.cuda()
    netYed.apply(weights_init_d)
    
    if opt.netYed != '':
        logging.info('-' * 8)
        logging.info('Load state_dict in {}'.format(opt.netYed))
        logging.info('-' * 8)
        netYed.load_state_dict(torch.load(opt.netYed))
    # print(netYed)


    criterion = nn.CrossEntropyLoss().cuda()
    

    # real_label = 0
    # fake_label = 1
    # setup optimizer
    optimizerD1 = optim.Adam(netXu.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
    optimizerD2 = optim.Adam(netYed.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))

    scheduler_D1 = StepLR(optimizerD1, step_size=20, gamma=0.4)
    scheduler_D2 = StepLR(optimizerD2, step_size=20, gamma=0.4)
    scheduler_G = StepLR(optimizerG, step_size=20, gamma=0.4)

    iteration_xu = 0
    iteration_yed = 0
    
    train_hist = {}
    train_hist['D_loss_Xu'] = []
    train_hist['D_loss_Yed'] = []
    train_hist['G_loss'] = []
    train_hist['per_epoch_time'] = []
    train_hist['total_time'] = []

    mse_loss = torch.nn.MSELoss()

    start_time = time.time()
    for epoch in range(opt.niter):

        netG.train()
        netXu.train()
        netYed.train()

        scheduler_D1.step()
        scheduler_D2.step()
        scheduler_G.step()


        epoch_start_time = time.time()
        for i,(sample,sample_flu) in enumerate(dataloader,0):

            data, rand_ud, label = sample['data'], sample['rand_ud'],sample['label']
            


            data_flu = sample_flu['data']
            
            # threshold
            if mse_loss(data,data_flu) > 16:
                continue



            cover, n, label = data.cuda(), rand_ud.cuda(), label.cuda()
            data_flu = data_flu.cuda()
            
            # train with real
            optimizerG.zero_grad()

            
            

            p = netG(cover)

            
            p_plus = p/2.0 + 1e-5
            p_minus = p/2.0 + 1e-5
            
            m =  - 0.5 * torch.tanh(TS_PARA * (p - 2 * n)) + 0.5 * torch.tanh(TS_PARA * (p - 2 * (1 - n))) 
            
            stego = cover + m

            C = -(p_plus * torch.log2(p_plus) + p_minus*torch.log2(p_minus)+ (1 - p+1e-5) * torch.log2(1 - p+1e-5))

            d_input = torch.zeros(opt.batchSize*2,1,512,512).cuda()
            d_input[0:opt.batchSize*2:2,:] = data
            d_input[1:opt.batchSize*2:2,:] = stego

            d_inputflu = torch.zeros(opt.batchSize*2,1,512,512).cuda()
            d_inputflu[0:opt.batchSize*2:2,:] = data_flu
            d_inputflu[1:opt.batchSize*2:2,:] = stego
            
        
            
            label = label.reshape(-1)


            
            optimizerD1.zero_grad()
            
            errD1 = criterion(netXu(d_inputflu.detach()), label)
            
            

            
            
            optimizerD2.zero_grad()
            
            
            errD2 = criterion(netYed(d_input.detach()), label)
            
   


            if  errD1.item() > errD2.item(): 
                errD1.backward()
                         
                optimizerD1.step()
           
                iteration_yed += 1


            else:
                errD2.backward()                
                optimizerD2.step()
                iteration_xu += 1

            
            
            
                
           
            g_l1 = criterion(netYed(d_input), label)*mylambda + criterion(netXu(d_inputflu), label)   


            g_l2 = torch.mean((C.sum(dim = (1,2,3)) - 512 * 512 * PAYLOAD) ** 2)
            
            
            errG = -g_l1 + 1e-7*g_l2
            if epoch > 0:
                train_hist['G_loss'].append(errG.item())
                train_hist['D_loss_Xu'].append(errD1.item())
                train_hist['D_loss_Yed'].append(errD2.item())

            
            errG.backward()
            optimizerG.step()

            logging.info('Epoch: [%d/%d][%d/%d] Loss_D1: %.4f Loss_D2: %.4f Loss_G: %.4f  C:%.4f' % (epoch, opt.niter-1, i, len(dataloader), errD1.item(), errD2.item(), errG.item() ,C.sum().item()/opt.batchSize))
            
        train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            
        # do checkpointing
        if (epoch+1)%10 == 2 and (epoch + 1) >= (opt.niter - 14):
        
        
            torch.save(netG.state_dict(), '%s/netG_epoch_%s_%d.pth' % (opt.outf, opt.config, epoch+1))
        
        logging.info("xunet for %d iters, YedNet for %d iters" % (iteration_xu, iteration_yed) )
        loss_plot(train_hist, opt.outf, model_name = opt.outf + opt.config)

    train_hist['total_time'].append(time.time() - start_time)
    logging.info("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(train_hist['per_epoch_time']),
                                                                            epoch, train_hist['total_time'][0]))
    logging.info("xunet for %d iters, YedNet for %d iters" % (iteration_xu, iteration_yed) )
    logging.info("Training finish!... save training results")

if __name__ == '__main__':
    main()