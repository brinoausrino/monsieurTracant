#!/usr/bin/python3

import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from .model import Generator,GlobalGenerator2,InceptionV3

#from model import Generator, GlobalGenerator2, InceptionV3
from .dataset import UnpairedDepthDataset
from PIL import Image
import numpy as np
from .utils import channel2width


class Options:
    def __init__(self):
        self.name = "contour_style"
        self.checkpoints_dir = "informativeDrawings/checkpoints"
        self.results_dir = "informativeDrawings/results"
        self.geom_name = "feats2Geom"
        self.batchSize=1
        self.dataroot = ""
        self.depthroot = ""

        self.input_nc=3
        self.output_nc=1
        self.geom_nc=3
        self.every_feat = 1
        self.num_classes = 55
        self.midas = 0

        self.ngf = 64
        self.n_blocks = 3
        self.size = 256
        self.cuda =  True
        self.n_cpu = 8
        self.which_epoch = "latest"
        self.aspect_ratio = 1.0

        self.mode = "test"
        self.load_size = 256
        self.crop_size = 256
        self.max_dataset_size = float("inf")
        self.preprocess = "resize_and_crop"
        self.no_flip = True
        self.norm = "instance"

        self.predict_depth = 0
        self.save_input = 0
        self.reconstruct = 0
        self.how_many = 100


net_G = 0
net_GB = 0
netGeom = 0

def init_nn(opt):
    global net_G
    global net_GB
    global netGeom

    opt.no_flip = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    with torch.no_grad():
        # Networks

        net_G = 0
        net_G = Generator(opt.input_nc, opt.output_nc, opt.n_blocks)
        net_G.cuda()

        net_GB = 0
        if opt.reconstruct == 1:
            net_GB = Generator(opt.output_nc, opt.input_nc, opt.n_blocks)
            net_GB.cuda()
            net_GB.load_state_dict(torch.load(os.path.join(opt.checkpoints_dir, opt.name, 'netG_B_%s.pth' % opt.which_epoch)))
            net_GB.eval()

        netGeom = 0
        if opt.predict_depth == 1:
            usename = opt.name
            if (len(opt.geom_name) > 0) and (os.path.exists(os.path.join(opt.checkpoints_dir, opt.geom_name))):
                usename = opt.geom_name
            myname = os.path.join(opt.checkpoints_dir, usename, 'netGeom_%s.pth' % opt.which_epoch)
            netGeom = GlobalGenerator2(768, opt.geom_nc, n_downsampling=1, n_UPsampling=3)

            netGeom.load_state_dict(torch.load(myname))
            netGeom.cuda()
            netGeom.eval()

            # numclasses = opt.num_classes
            ### load pretrained inception
            net_recog = InceptionV3(opt.num_classes, False, use_aux=True, pretrain=True, freeze=True, every_feat=opt.every_feat==1)
            net_recog.cuda()
            net_recog.eval()

        # Load state dicts
        net_G.load_state_dict(torch.load(os.path.join(opt.checkpoints_dir, opt.name, 'netG_A_%s.pth' % opt.which_epoch)))
        print('loaded', os.path.join(opt.checkpoints_dir, opt.name, 'netG_A_%s.pth' % opt.which_epoch))

        # Set model's test mode
        net_G.eval()



def proceed_conversion(opt):
    global net_G
    global net_GB
    global netGeom

    with torch.no_grad():
        transforms_r = [transforms.Resize(int(opt.size), Image.BICUBIC),
                    transforms.ToTensor()]
        test_data = UnpairedDepthDataset(opt.dataroot, '', opt, transforms_r=transforms_r, 
                    mode=opt.mode, midas=opt.midas>0, depthroot=opt.depthroot)

        dataloader = DataLoader(test_data, batch_size=opt.batchSize, shuffle=False)

        ###################################

        ###### Testing######

        full_output_dir = os.path.join(opt.results_dir, opt.name)

        if not os.path.exists(full_output_dir):
            os.makedirs(full_output_dir)

        for i, batch in enumerate(dataloader):
            if i > opt.how_many:
                break;
            img_r  = Variable(batch['r']).cuda()
            img_depth  = Variable(batch['depth']).cuda()
            real_A = img_r

            name = batch['name'][0]
            
            input_image = real_A
            image = net_G(input_image)
            save_image(image.data, full_output_dir+'/%s_out.png' % name)

            if (opt.predict_depth == 1):

                geom_input = image
                if geom_input.size()[1] == 1:
                    geom_input = geom_input.repeat(1, 3, 1, 1)
                _, geom_input = net_recog(geom_input)
                geom = netGeom(geom_input)
                geom = (geom+1)/2.0 ###[-1, 1] ---> [0, 1]

                input_img_fake = channel2width(geom)
                save_image(input_img_fake.data, full_output_dir+'/%s_geom.png' % name)

            if opt.reconstruct == 1:
                rec = net_GB(image)
                save_image(rec.data, full_output_dir+'/%s_rec.png' % name)

            if opt.save_input == 1:
                save_image(img_r, full_output_dir+'/%s_input.png' % name)

            sys.stdout.write('\rGenerated images %04d of %04d' % (i, opt.how_many))

        sys.stdout.write('\n')
        ###################################

def proceed_conversion_folder(opt):
    
    opt.no_flip = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    with torch.no_grad():
        # Networks

        net_G = 0
        net_G = Generator(opt.input_nc, opt.output_nc, opt.n_blocks)
        net_G.cuda()

        net_GB = 0
        if opt.reconstruct == 1:
            net_GB = Generator(opt.output_nc, opt.input_nc, opt.n_blocks)
            net_GB.cuda()
            net_GB.load_state_dict(torch.load(os.path.join(opt.checkpoints_dir, opt.name, 'netG_B_%s.pth' % opt.which_epoch)))
            net_GB.eval()
        
        netGeom = 0
        if opt.predict_depth == 1:
            usename = opt.name
            if (len(opt.geom_name) > 0) and (os.path.exists(os.path.join(opt.checkpoints_dir, opt.geom_name))):
                usename = opt.geom_name
            myname = os.path.join(opt.checkpoints_dir, usename, 'netGeom_%s.pth' % opt.which_epoch)
            netGeom = GlobalGenerator2(768, opt.geom_nc, n_downsampling=1, n_UPsampling=3)

            netGeom.load_state_dict(torch.load(myname))
            netGeom.cuda()
            netGeom.eval()

            numclasses = opt.num_classes
            ### load pretrained inception
            net_recog = InceptionV3(opt.num_classes, False, use_aux=True, pretrain=True, freeze=True, every_feat=opt.every_feat==1)
            net_recog.cuda()
            net_recog.eval()

        # Load state dicts
        net_G.load_state_dict(torch.load(os.path.join(opt.checkpoints_dir, opt.name, 'netG_A_%s.pth' % opt.which_epoch)))
        print('loaded', os.path.join(opt.checkpoints_dir, opt.name, 'netG_A_%s.pth' % opt.which_epoch))

        # Set model's test mode
        net_G.eval()

        
        transforms_r = [transforms.Resize(int(opt.size), Image.BICUBIC),
                    transforms.ToTensor()]


        test_data = UnpairedDepthDataset(opt.dataroot, '', opt, transforms_r=transforms_r, 
                    mode=opt.mode, midas=opt.midas>0, depthroot=opt.depthroot)

        dataloader = DataLoader(test_data, batch_size=opt.batchSize, shuffle=False)

        ###################################

        ###### Testing######

        full_output_dir = os.path.join(opt.results_dir, opt.name)

        if not os.path.exists(full_output_dir):
            os.makedirs(full_output_dir)

        for i, batch in enumerate(dataloader):
            if i > opt.how_many:
                break;
            img_r  = Variable(batch['r']).cuda()
            img_depth  = Variable(batch['depth']).cuda()
            real_A = img_r

            name = batch['name'][0]
            
            input_image = real_A
            image = net_G(input_image)
            save_image(image.data, full_output_dir+'/%s_out.png' % name)

            if (opt.predict_depth == 1):

                geom_input = image
                if geom_input.size()[1] == 1:
                    geom_input = geom_input.repeat(1, 3, 1, 1)
                _, geom_input = net_recog(geom_input)
                geom = netGeom(geom_input)
                geom = (geom+1)/2.0 ###[-1, 1] ---> [0, 1]

                input_img_fake = channel2width(geom)
                save_image(input_img_fake.data, full_output_dir+'/%s_geom.png' % name)

            if opt.reconstruct == 1:
                rec = net_GB(image)
                save_image(rec.data, full_output_dir+'/%s_rec.png' % name)

            if opt.save_input == 1:
                save_image(img_r, full_output_dir+'/%s_input.png' % name)

            sys.stdout.write('\rGenerated images %04d of %04d' % (i, opt.how_many))

        sys.stdout.write('\n')
        ###################################
