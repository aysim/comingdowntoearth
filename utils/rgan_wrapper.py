import os
from utils.base_wrapper import BaseModel
from networks.c_gan import *

class RGANWrapper(BaseModel):

    def __init__(self, opt, log_file, net_G, net_D, net_R):
        BaseModel.__init__(self, opt, log_file)
        self.optimizers = []
        self.ret_best_acc = 0.0
        #initialize generator, discriminator and retrieval
        self.generator = net_G
        self.retrieval = net_R
        self.discriminator = net_D

        self.criterion = GANLoss(opt.gan_loss).to(opt.device)
        self.criterion_l1 = torch.nn.L1Loss()

        # Optimizers
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=opt.lr_d, betas=(opt.b1, opt.b2))
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=opt.lr_g, betas=(opt.b1, opt.b2))
        self.optimizer_R = torch.optim.Adam(self.retrieval.parameters(), lr=opt.lr_r, betas=(opt.b1, opt.b2))

        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)
        self.optimizers.append(self.optimizer_R)
        self.load_networks()

    def forward(self):
        self.fake_street, self.residual = self.generator(self.satellite)
    def backward_D(self):

        # Fake discriminator train
        fake_satellitestreet = torch.cat((self.satellite, self.fake_street), 1)
        fake_pred = self.discriminator(fake_satellitestreet.detach())
        self.d_loss_fake = self.criterion(fake_pred, False)

        # Real discriminator train
        real_satellitestreet = torch.cat((self.satellite, self.street), 1)
        real_pred = self.discriminator(real_satellitestreet)
        self.d_loss_real = self.criterion(real_pred, True)

        self.d_loss = (self.d_loss_real + self.d_loss_fake) * 0.5
        self.d_loss.backward()


    def backward_R(self, epoch):

        self.fake_street_out, self.street_out = self.retrieval(self.street, self.residual.detach())
        if epoch <= 20:
            self.r_loss = self.soft_margin_triplet_loss(self.fake_street_out, self.street_out, loss_weight=self.opt.lambda_sm, hard_topk_ratio=self.opt.hard_topk_ratio)
        elif epoch > 20 and epoch <=40:
            self.r_loss = self.soft_margin_triplet_loss(self.fake_street_out, self.street_out, loss_weight=self.opt.lambda_sm, hard_topk_ratio=self.opt.hard_decay1_topk_ratio)
        elif epoch > 40 and epoch <=60:
            self.r_loss = self.soft_margin_triplet_loss(self.fake_street_out, self.street_out, loss_weight=self.opt.lambda_sm, hard_topk_ratio=self.opt.hard_decay2_topk_ratio)
        elif epoch > 60:
            self.r_loss = self.soft_margin_triplet_loss(self.fake_street_out, self.street_out, loss_weight=self.opt.lambda_sm, hard_topk_ratio=self.opt.hard_decay3_topk_ratio)
        self.r_loss.backward()


    def backward_G(self, epoch):

        fake_satellitestreet = torch.cat((self.satellite, self.fake_street), 1)
        fake_pred = self.discriminator(fake_satellitestreet)
        self.gan_loss = self.criterion(fake_pred, True)

        self.fake_street_out, self.street_out = self.retrieval(self.street, self.residual)

        if epoch <= 20:
            self.ret_loss = self.soft_margin_triplet_loss(self.fake_street_out, self.street_out, loss_weight=self.opt.lambda_sm,
                                                        hard_topk_ratio=self.opt.hard_topk_ratio) * self.opt.lambda_ret1
        elif epoch > 20 and epoch <=40:
            self.ret_loss = self.soft_margin_triplet_loss(self.fake_street_out, self.street_out,
                                                          loss_weight=self.opt.lambda_sm,
                                                          hard_topk_ratio=self.opt.hard_decay1_topk_ratio) * self.opt.lambda_ret1
        elif epoch > 40 and epoch <=60:
            self.ret_loss = self.soft_margin_triplet_loss(self.fake_street_out, self.street_out,
                                                          loss_weight=self.opt.lambda_sm,
                                                          hard_topk_ratio=self.opt.hard_decay2_topk_ratio) * self.opt.lambda_ret1
        elif epoch > 60:
            self.ret_loss = self.soft_margin_triplet_loss(self.fake_street_out, self.street_out,
                                                          loss_weight=self.opt.lambda_sm,
                                                          hard_topk_ratio=self.opt.hard_decay3_topk_ratio) * self.opt.lambda_ret1

        self.g_l1 = self.criterion_l1(self.fake_street, self.street) * self.opt.lambda_l1
        self.g_loss = self.ret_loss + self.gan_loss + self.g_l1
        self.g_loss.backward()

    def optimize_parameters(self, epoch):

        self.forward()

        # update D
        self.set_requires_grad(self.discriminator, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update R
        self.set_requires_grad(self.discriminator, False)
        self.set_requires_grad(self.retrieval, True)
        self.optimizer_R.zero_grad()
        self.backward_R(epoch)
        self.optimizer_R.step()

        # update G
        self.set_requires_grad(self.retrieval, False)
        self.optimizer_G.zero_grad()
        self.backward_G(epoch)
        self.optimizer_G.step()

    def eval_model(self):
        self.forward()
        self.fake_street_out_val, self.street_out_val = self.retrieval(self.street, self.residual)


    def save_networks(self, epoch, out_dir, last_ckpt=False, best_acc=None, is_best=False):
        ckpt = {'last_epoch': epoch,
                'best_acc': best_acc,
                'generator_model_dict': self.generator.state_dict(),
                'optimizer_G_dict': self.optimizer_G.state_dict(),
                'discriminator_model_dict': self.discriminator.state_dict(),
                'optimizer_D_dict': self.optimizer_D.state_dict(),
                'retriebal_model_dict': self.retrieval.state_dict(),
                 'optimizer_R_dict': self.optimizer_R.state_dict(),
                }

        if last_ckpt:
            ckpt_name = 'rgan_last_ckpt.pth'
        elif is_best:
            ckpt_name = 'rgan_best_ckpt.pth'
        else:
            ckpt_name = 'rgan_ckpt_ep{}.pth'.format(epoch + 1)
        ckpt_path = os.path.join(out_dir, ckpt_name)
        torch.save(ckpt, ckpt_path)

    def load_networks(self):
        if self.opt.rgan_checkpoint is None:
            return

        ckpt_path = self.opt.rgan_checkpoint
        ckpt = torch.load(ckpt_path)

        self.opt.start_epoch = ckpt['last_epoch'] + 1
        self.ret_best_acc = ckpt['best_acc']

        # Load net state
        generator_dict = ckpt['generator_model_dict']
        discriminator_dict = ckpt['discriminator_model_dict']
        retrieval_dict = ckpt['retriebal_model_dict']

        self.generator.load_state_dict(generator_dict, strict=False)
        self.optimizer_G = ckpt['optimizer_G_dict']

        self.discriminator.load_state_dict(discriminator_dict)
        self.optimizer_D = ckpt['optimizer_D_dict']

        self.retrieval.load_state_dict(retrieval_dict)
        self.optimizer_R = ckpt['optimizer_R_dict']
