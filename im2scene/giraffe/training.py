from im2scene.eval import (
    calculate_activation_statistics, calculate_frechet_distance)
from im2scene.training import (
    toggle_grad, compute_grad2, compute_bce, update_average, make_patchgan_target, cal_gradient_penalty, wgan_loss)
from torchvision.utils import save_image, make_grid
import os
import torch
import torch.nn as nn
from im2scene.training import BaseTrainer
from tqdm import tqdm
import logging
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np 
import pdb 
import math 
import pytorch_msssim
from .metrics import PSNR, SSIM, LPIPS

logger_py = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    ''' Trainer object for GIRAFFE.

    Args:
        model (nn.Module): GIRAFFE model
        optimizer (optimizer): generator optimizer object
        optimizer_d (optimizer): discriminator optimizer object
        device (device): pytorch device
        vis_dir (str): visualization directory
        multi_gpu (bool): whether to use multiple GPUs for training
        fid_dict (dict): dicionary with GT statistics for FID
        n_eval_iterations (int): number of eval iterations
        overwrite_visualization (bool): whether to overwrite
            the visualization files
    '''

    def __init__(self, model, optimizer, optimizer_d, device=None,
                 vis_dir=None,
                 val_vis_dir=None,
                 multi_gpu=False, fid_dict={},
                 n_eval_iterations=10,
                 overwrite_visualization=True, batch_size=None, recon_weight=None, cam_weight = None, **kwargs):

        self.model = model
        self.optimizer = optimizer
        self.optimizer_d = optimizer_d
        self.device = device
        self.vis_dir = vis_dir
        self.val_vis_dir = val_vis_dir
        self.multi_gpu = multi_gpu
        self.m = pytorch_msssim.MSSSIM()

        self.overwrite_visualization = overwrite_visualization
        self.fid_dict = fid_dict
        self.n_eval_iterations = n_eval_iterations
        self.recon_loss = torch.nn.MSELoss()
        self.vis_dict = model.generator.get_vis_dict(batch_size)
        self.recon_weight = recon_weight
        self.cam_weight = cam_weight

        self.psnr = PSNR(float)
        self.ssim = SSIM(float)
        self.lpips = LPIPS(float)


        if multi_gpu:
            self.generator = torch.nn.DataParallel(self.model.generator)
            self.discriminator = torch.nn.DataParallel(
                self.model.discriminator)
            if self.model.generator_test is not None:
                self.generator_test = torch.nn.DataParallel(
                    self.model.generator_test)
            else:
                self.generator_test = None
        else:
            self.generator = self.model.generator
            self.discriminator = self.model.discriminator
            self.generator_test = self.model.generator_test

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

        if val_vis_dir is not None and not os.path.exists(val_vis_dir):
            os.makedirs(val_vis_dir)


    def train_step(self, data, it=None):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
            it (int): training iteration
        '''
        loss_gen, cam_pred_loss, shape_loss, appearance_loss, recon_loss = self.train_step_generator(data, it)

        return {
            'gen_total': loss_gen,
            'shape_loss': shape_loss,
            'app_loss': appearance_loss,
            'cam_loss': cam_pred_loss,
            'recon_loss': recon_loss,
        }



    def eval_step(self, data):
        ''' Performs a validation step.

        Args:
            data (dict): data dictionary
        '''
        gen = self.model.generator_test
        if gen is None:

            gen = self.model.generator
        gen.eval()


        x_real = data.get('image').to(self.device)
        pose_real = data.get('pose').to(self.device)

        self.generator.eval()
        with torch.no_grad():
            x_fake = self.generator(x_real, pose_real, mode='val')[0].cpu()[:, :3]       # new_rgb가 결국 최종 대상! <- predictor와 NeRF 둘다 잘 학습됐는지!
        self.generator.train()

        np_fake, np_real = x_fake.permute(0, 2, 3, 1).numpy().astype(np.float), x_real.permute(0, 2, 3, 1).cpu().numpy().astype(np.float)
        psnr_total = 0
        ssim_total = 0
        total = 0

        for x, y in zip(np_fake, np_real):
            psnr_total += self.psnr(x, y)
            ssim_total += self.ssim(x, y)
            total += 1

        psnrAE = psnr_total / total, 
        ssimAE = ssim_total / total, 
        lpipsAE = self.lpips(x_fake * 2 - 1, x_real * 2 - 1).mean().item()


        eval_dict = {
            'psnrAE': psnrAE, 
            'ssimAE': ssimAE,
            'lpipsAE': lpipsAE
        }
        return eval_dict




    def train_step_generator(self, data, it=None, z=None):
        generator = self.generator
        discriminator = self.discriminator

        toggle_grad(generator, True)
        toggle_grad(discriminator, False)

        generator.train()
        discriminator.train()

        self.optimizer.zero_grad()

        pair_img = data.get('image').to(self.device)
        pose_real = data.get('pose').to(self.device)
        # pair_shape = data.get('shape').to(self.device)
        # pair_appearance = data.get('appearance').to(self.device)

        # find gt_trans
        gt_trans = pose_real[:, :3, -1]
        # proj w/o trans into scale, rotation은 여기서는 필요없기 때문에 패스! -> numpy, pytorch를 왔다갔다해야하기 때문에 시간 많이 잡아먹음 
        gt_rot = pose_real[:, :3, :3]
        gt_scale = torch.tensor([1.]).reshape(-1, 1).repeat(len(gt_rot), 1).to(self.device)
        
        if self.multi_gpu:
            latents = generator.module.get_vis_dict(x_real.shape[0])
            pred_img, pred_cam, latent_codes = generator(pair_img, pose_real, **latents)        # pred, swap, rand

        else:
            pred_img, pred_cam, latent_codes = generator(pair_img, pose_real)

        
        # cam GT loss 주기 
        rot_from_fake = pred_cam[-1]
        pred_trans_loss = torch.norm(pred_cam[1] - gt_trans)
        pred_scale_loss = torch.norm(pred_cam[0] - gt_scale)
        pred_rot_loss = torch.norm(torch.bmm(torch.inverse(rot_from_fake), gt_rot) - torch.eye(3).reshape(-1, 3, 3).repeat(len(gt_rot), 1, 1).to(self.device))

        cam_pred_loss = pred_scale_loss + pred_trans_loss + pred_rot_loss

        # shape_loss = torch.norm(latent_codes[0] - pair_shape)
        # app_loss = torch.norm(latent_codes[1] - pair_appearance)

        shape_loss, app_loss = cam_pred_loss * 2, cam_pred_loss * 2

        recon_loss = self.recon_loss(pred_img, pair_img) * self.recon_weight
        gen_loss = cam_pred_loss + shape_loss + app_loss + recon_loss

        gen_loss.backward()     # computational graph를 끊김, gradient 쌓기
        self.optimizer.step()   # 모델 파라미터에 쌓인 graident가 업데이트  

        if self.generator_test is not None:
            update_average(self.generator_test, generator, beta=0.999)

        return gen_loss.item(), cam_pred_loss.item(), shape_loss.item(), app_loss.item(), recon_loss.item()


    def train_step_discriminator(self, data, it=None, z=None):
        generator = self.generator
        discriminator = self.discriminator
        toggle_grad(generator, False)
        toggle_grad(discriminator, True)
        # 얘네 왜함..? -> train모드로..? 
        generator.train()
        discriminator.train()
        patchgan_loss = nn.MSELoss()

        self.optimizer_d.zero_grad()

        x_real = data.get('image').to(self.device)
        pose_real = data.get('pose').to(self.device)

        loss_d_full = 0.

        x_real.requires_grad_()
        d_real = discriminator(x_real)

        d_real_target = make_patchgan_target(d_real, need_true = True) #lsgan
        d_loss_real = patchgan_loss(d_real, d_real_target)

        loss_d_full += d_loss_real 

        reg = compute_grad2(d_real, x_real).mean() * 10.
        loss_d_full += reg

        with torch.no_grad():
            if self.multi_gpu:
                latents = generator.module.get_vis_dict(x_real.shape[0])
                x_fake, pred_cam = generator(x_real, pose_real, **latents)        # pred, swap, rand

            else:
                x_fake, pred_cam = generator(x_real, pose_real)

        x_fake.requires_grad_()
        d_fake = discriminator(x_fake)
        d_fake_target = make_patchgan_target(d_fake, need_true = False)

        d_loss_fake = patchgan_loss(d_fake, d_fake_target)
        loss_d_full += d_loss_fake

        loss_d_full.backward()
        self.optimizer_d.step()

        d_loss = (d_loss_real + d_loss_fake)

        return (
            d_loss.item(), reg.item(), d_loss_real.item(), d_loss_fake.item())




    def record_uvs(self, uv, path, it):
        out_path = os.path.join(path, 'uv.txt')
        name_dict = {0: 'pred_GT', 1: 'GT', 2: 'swap_GT', 3:'rand'}
        if not os.path.exists(out_path):
            f = open(out_path, 'w')
        else:
            f = open(out_path, 'a')

        uvs = torch.stack(uv, dim=0)
        for i in range(len(uvs)):        # len: 3
            line = list(map(lambda x: round(x, 3), uvs[i].flatten().detach().cpu().numpy().tolist()))
            out = []
            for idx in range(0, len(line)//2):
                out.append(tuple((line[2*idx], line[2*idx+1])))
            txt_line = f'{it}th {name_dict[i]}-uv : {out}\n'
            f.write(txt_line)
        f.write('\n')
        f.close()

    def uv2rad(self, uv):
        theta = 360 * uv[:, 0]
        phi = torch.arccos(1 - 2 * uv[:, 1]) / math.pi * 180
        
        return torch.stack([theta, phi], dim=-1) #16, 2

    def loc2rad(self, loc):
        phi = torch.acos(loc[:, 2])
        plusidx = torch.where(loc[:,1]<0)[0]
        theta = torch.acos(loc[:,0]/torch.sin(phi))
        theta[plusidx] = 2*math.pi-theta[plusidx]
        theta, phi = theta * 180 / torch.pi, phi * 180 / torch.pi
        return torch.cat([theta, phi], dim=-1)


    def visualize(self, data, it=0, mode=None, val_idx=None):
        ''' Visualized the data.

        Args:
            it (int): training iteration
        '''
        gen = self.model.generator_test
        if gen is None:
            gen = self.model.generator
        gen.eval()
        with torch.no_grad():
            # edit mira start 
            x_real = data.get('image').to(self.device)
            x_pose = data.get('pose').to(self.device)

            rgb, shape_rgb, app_rgb, pose_rgb, pred_cam = self.generator(x_real, x_pose, mode='val')
            rgb, shape_rgb, app_rgb, pose_rgb = rgb.detach(), shape_rgb.detach(), app_rgb.detach(), pose_rgb.detach()

            '''
            # 여기 안을 잘 조정하면 -> swapped view에서도 비슷한 맥락으로 나올 듯 
            # edit 'ㅅ'
            randrad = self.uv2rad(uvs).cuda()        # uv를 radian으로 표현 
            rotmat1 = x_pose[:,:3,:3]        # x_pose : real pose that includes R,t     # rotation matrix 
            origin = torch.Tensor([0,0,1]).to(self.device).repeat(int(len(x_pose)),1).unsqueeze(-1)

            camloc1 = rotmat1@origin    # rotmat1@origin의 norm은 1, translation까지 더해줘야 함 
            radian1 = self.loc2rad(camloc1) 

            camloc_new = pred_cam[-1]@origin    # rotmat1@origin의 norm은 1, translation까지 더해줘야 함 
            radian_new = self.loc2rad(camloc_new ) 

            uvs_full = (radian_new, radian1, radian1.flip(0), randrad)#gt, swap    3, 16, 2
            '''

        out_file_name = 'visualization_%010d.png' % it
        # edit mira end                                     
        image_grid = make_grid(torch.cat((x_real.to(self.device), rgb.clamp_(0., 1.), shape_rgb.clamp_(0., 1.), app_rgb.clamp_(0., 1.), pose_rgb.clamp_(0., 1.)), dim=0), nrow=pose_rgb.shape[0])
        save_image(image_grid, os.path.join(self.vis_dir, out_file_name))
        # self.record_uvs(uvs_full, os.path.join(self.vis_dir), it)
        gen.train()

        return image_grid