import argparse
import os
import torch

class Parser():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        #basic parameters
        parser.add_argument('--results_dir', '-o', type=str, default='./placeholder_results_dir', help='models are saved here')
        parser.add_argument('--name', type=str, default='', help=' ')
        parser.add_argument('--seed', type=int, default=10)
        parser.add_argument('--phase', type=str, default='train', help='')
        parser.add_argument('--gpu_ids', type=str, default='0, 1', help='gpu ids: e.g. 0  0,1')

        parser.add_argument('--isTrain', default=True, action='store_true')
        parser.add_argument('--resume', default=True, action='store_true')
        parser.add_argument('--start_epoch', type=int, default=0)

        #data parameters
        parser.add_argument('--data_root', type=str, default= './placeholder_data_path')
        parser.add_argument('--train_csv', type=str, default='train-19zl.csv')
        parser.add_argument('--val_csv', type=str, default='val-19zl.csv')
        parser.add_argument('--polar', default=True, action='store_true')
        parser.add_argument('--save_step', type=int, default=10)

        parser.add_argument('--rgan_checkpoint', type=str, default=None)

        #train parameters
        parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of combined training")
        parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
        parser.add_argument("--lr_g", type=float, default=0.0001, help="adam: learning rate")
        parser.add_argument("--lr_d", type=float, default=0.0001, help="adam: learning rate")
        parser.add_argument("--lr_r", type=float, default=0.0001, help="adam: learning rate")

        parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
        parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

        #loss parameters
        parser.add_argument("--lambda_gp", type=int, default=10, help="loss weight for gradient penalty")
        parser.add_argument("--lambda_l1", type=int, default=100, help="loss weight for l1")
        parser.add_argument("--lambda_ret1", type=int, default=1000, help="loss weight for retrieval")
        parser.add_argument("--lambda_sm", type=int, default=10, help="loss weight for soft margin")
        parser.add_argument("--hard_topk_ratio", type=float, default=1.0, help="hard negative ratio")
        parser.add_argument("--hard_decay1_topk_ratio", type=float, default=0.1, help="hard negative ratio")
        parser.add_argument("--hard_decay2_topk_ratio", type=float, default=0.05, help="hard negative ratio")
        parser.add_argument("--hard_decay3_topk_ratio", type=float, default=0.01, help="hard negative ratio")

        #gan parameters
        parser.add_argument("--n_critic", type=int, default=1,
                            help="number of training steps for discriminator per iter")

        parser.add_argument("--input_c", type=int, default=3)
        parser.add_argument("--segout_c", type=int, default=3)
        parser.add_argument("--realout_c", type=int, default=3)
        parser.add_argument("--n_layers", type=int, default=3)
        parser.add_argument("--feature_c", type=int, default=64)
        parser.add_argument('--g_model', type=str, default='unet-skip')
        parser.add_argument('--d_model', type=str, default='basic')
        parser.add_argument('--r_model', type=str, default='SAFA')
        parser.add_argument('--gan_loss', type=str, default='vanilla')

        parser.add_argument("--lambda", type=int, default=10)
        parser.add_argument("--condition", type=int, default=1)

        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        # save and return the parser
        self.parser = parser
        return parser.parse_args()


    def print_options(self, opt):

        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        prefix = '{}_{}_lrg{}_lrd{}_lrr{}_batch{}_l1w{}_retl1w_{}_HN_{}_HN1decay_{}HN2decay_{}HN3decay_{}'.format(opt.g_model, opt.d_model, opt.lr_g,
                                                                                                         opt.lr_d, opt.lr_r,
                                                                                                         opt.batch_size, opt.lambda_l1, opt.lambda_ret1,
                                                                                                         opt.hard_topk_ratio, opt.hard_decay1_topk_ratio,
                                                                                                         opt.hard_decay2_topk_ratio, opt.hard_decay3_topk_ratio)

        out_dir = os.path.join(opt.results_dir, prefix)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        file_name = os.path.join(out_dir, 'log.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
            opt_file.flush()
        return file_name


    def parse(self):
        opt = self.gather_options()
        file = self.print_options(opt)
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt, file

    def log(self, ms, log=None):
        print(ms)
        if log:
            log.write(ms + '\n')
            log.flush()
