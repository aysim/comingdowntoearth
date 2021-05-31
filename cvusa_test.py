from data.custom_transforms import *
from data.utils import CVUSA
from networks.c_gan import *
from utils import rgan_wrapper
from utils.setup_helper import *
from argparse import Namespace
from helper import parser
import os
from matplotlib import pyplot


if __name__ == '__main__':
    parse = parser.Parser()
    opt, log_file = parse.parse()
    opt.is_Train = True
    make_deterministic(opt.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.gpu_ids)

    log = open(log_file, 'a')
    log_print = lambda ms: parse.log(ms, log)

    #define networks
    generator = define_G(netG=opt.g_model, gpu_ids=opt.gpu_ids)
    print('Init {} as generator model'.format(opt.g_model))

    discriminator = define_D(input_c=opt.input_c, output_c=opt.realout_c, ndf=opt.feature_c, netD=opt.d_model,
                             condition=opt.condition, n_layers_D=opt.n_layers, gpu_ids=opt.gpu_ids)
    print('Init {} as discriminator model'.format(opt.d_model))

    retrieval = define_R(ret_method=opt.r_model, polar=opt.polar, gpu_ids=opt.gpu_ids)
    print('Init {} as retrieval model'.format(opt.r_model))

    # Initialize network wrapper
    if opt.resume:
        opt.rgan_checkpoint = os.path.join('./placeholder_checkpoint_path', 'rgan_best_ckpt.pth')

    rgan_wrapper = rgan_wrapper.RGANWrapper(opt, log_file, generator, discriminator, retrieval)
    # Configure data loader
    val_dataset = CVUSA(root=opt.data_root, csv_file=opt.val_csv, use_polar=opt.polar, name=opt.name,
                        transform_op=ToTensor())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)

    log_print('Load test dataset from {}: val_set={}'.format(opt.data_root, len(val_dataset)))
    log_print('length of val loader: {:d}'.format(len(val_loader)))

    rgan_wrapper.generator.eval()
    rgan_wrapper.retrieval.eval()
    fake_street_batches_v = []
    street_batches_v = []
    item_ids = []

    for i, data in enumerate(val_loader):
        print (i)
        rgan_wrapper.set_input(data)
        rgan_wrapper.eval_model()
        fake_street_batches_v.append(rgan_wrapper.fake_street_out_val.cpu().data)
        street_batches_v.append(rgan_wrapper.street_out_val.cpu().data)

    fake_street_vec = torch.cat(fake_street_batches_v, dim=0)
    street_vec = torch.cat(street_batches_v, dim=0)
    dists = 2 - 2 * torch.matmul(fake_street_vec, street_vec.permute(1, 0))

    tp1 = rgan_wrapper.mutual_topk_acc(dists, topk=1)
    tp5 = rgan_wrapper.mutual_topk_acc(dists, topk=5)
    tp10 = rgan_wrapper.mutual_topk_acc(dists, topk=10)

    num = len(dists)
    tp1p = rgan_wrapper.mutual_topk_acc(dists, topk=0.01 * num)
    acc = Namespace(num=len(dists), tp1=tp1, tp5=tp5, tp10=tp10, tp1p=tp1p)

    log_print('\nEvaluate Samples:{num:d}\nRecall(p2s/s2p) tp1:{tp1[0]:.2f}/{tp1[1]:.2f} ' \
          'tp5:{tp5[0]:.2f}/{tp5[1]:.2f} tp10:{tp10[0]:.2f}/{tp10[1]:.2f} ' \
         'tp1%:{tp1p[0]:.2f}/{tp1p[1]:.2f}'.format(1, num=acc.num, tp1=acc.tp1,
                                                   tp5=acc.tp5, tp10=acc.tp10, tp1p=acc.tp1p))
