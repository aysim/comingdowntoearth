from helper import parser_cvact
from data.custom_transforms import *
from data.cvact_utils import CVACT
from networks.c_gan import *
from os.path import exists, join, basename, dirname
from utils import rgan_wrapper_cvact
from utils.setup_helper import *
import time
from argparse import Namespace
import os

if __name__ == '__main__':
    parse = parser_cvact.Parser()
    opt, log_file = parse.parse()
    opt.is_Train = True
    make_deterministic(opt.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.gpu_ids)
    log = open(log_file, 'a')
    log_print = lambda ms: parse.log(ms, log)

    #define networks
    generator = define_G(netG=opt.g_model, gpu_ids=opt.gpu_ids)
    log_print('Init {} as generator model'.format(opt.g_model))

    discriminator = define_D(input_c=opt.input_c, output_c=opt.realout_c, ndf=opt.feature_c, netD=opt.d_model,
                             condition=opt.condition, n_layers_D=opt.n_layers, gpu_ids=opt.gpu_ids)
    log_print('Init {} as discriminator model'.format(opt.d_model))

    retrieval = define_R(ret_method=opt.r_model, polar=opt.polar, gpu_ids=opt.gpu_ids)
    log_print('Init {} as retrieval model'.format(opt.r_model))
    rgan_wrapper = rgan_wrapper_cvact.RGANWrapper(opt, log_file, generator, discriminator, retrieval)

    # Configure data loader
    composed_transforms = transforms.Compose([RandomHorizontalFlip(),
                                              ToTensor()])
    train_dataset = CVACT(root= opt.data_root, all_data_list = opt.data_list, use_polar=opt.polar, isTrain=opt.isTrain, transform_op=composed_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)

    val_dataset = CVACT(root= opt.data_root, all_data_list = opt.data_list, use_polar=opt.polar, isTrain=False, transform_op=ToTensor())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)

    log_print(
        'Load datasets from {}: train_set={} val_set={}'.format(opt.data_root, len(train_dataset), len(val_dataset)))

    ret_best_acc = rgan_wrapper.ret_best_acc
    log_print('Start training from epoch {} to {}, best acc: {}'.format(opt.start_epoch, opt.n_epochs, ret_best_acc))
    for epoch in range(opt.start_epoch, opt.n_epochs):
        start_time = time.time()
        batches_done = 0
        val_batches_done = 0
        street_batches_t = []
        fake_street_batches_t = []
        street_batches_v = []
        fake_street_batches_v = []
        epoch_retrieval_loss = []
        epoch_generator_loss = []
        epoch_discriminator_loss = []
        log_print('>>> RGAN Epoch {}'.format(epoch))
        rgan_wrapper.generator.train()
        rgan_wrapper.discriminator.train()
        rgan_wrapper.retrieval.train()
        for i, (data, utm) in enumerate(train_loader):  # inner loop within one epoch

            rgan_wrapper.set_input_cvact(data, utm)
            rgan_wrapper.optimize_parameters(epoch)

            fake_street_batches_t.append(rgan_wrapper.fake_street_out.cpu().data)
            street_batches_t.append(rgan_wrapper.street_out.cpu().data)
            epoch_retrieval_loss.append(rgan_wrapper.r_loss.item())
            epoch_discriminator_loss.append(rgan_wrapper.d_loss.item())
            epoch_generator_loss.append(rgan_wrapper.g_loss.item())

            if (i + 1) % 40 == 0 or (i + 1) == len(train_loader):
                fake_street_vec = torch.cat(fake_street_batches_t, dim=0)
                street_vec = torch.cat(street_batches_t, dim=0)
                dists = 2 - 2 * torch.matmul(fake_street_vec, street_vec.permute(1, 0))
                tp1 = rgan_wrapper.mutual_topk_acc(dists, topk=1)
                tp5 = rgan_wrapper.mutual_topk_acc(dists, topk=5)
                tp10 = rgan_wrapper.mutual_topk_acc(dists, topk=10)
                log_print('Batch:{} loss={:.3f} samples:{} tp1={tp1[0]:.2f}/{tp1[1]:.2f} ' \
                        'tp5={tp5[0]:.2f}/{tp5[1]:.2f}'.format(i + 1, np.mean(epoch_retrieval_loss),
                                                               len(dists), tp1=tp1, tp5=tp5))
                street_batches_t.clear()
                fake_street_batches_t.clear()

        rgan_wrapper.save_networks(epoch, dirname(log_file), best_acc=ret_best_acc,
                                        last_ckpt=True)  # Always save last ckpt


        # Save model periodically
        if (epoch + 1) % opt.save_step == 0:
            rgan_wrapper.save_networks(epoch, dirname(log_file), best_acc=ret_best_acc)

        rgan_wrapper.generator.eval()
        rgan_wrapper.retrieval.eval()
        for i, (data, utm) in enumerate(val_loader):
            rgan_wrapper.set_input_cvact(data, utm)
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
                    'tp1%:{tp1p[0]:.2f}/{tp1p[1]:.2f}'.format(epoch + 1, num=acc.num, tp1=acc.tp1,
                                                            tp5=acc.tp5, tp10=acc.tp10, tp1p=acc.tp1p))

        # Save the best model
        tp1_p2s_acc = acc.tp1[0]
        if tp1_p2s_acc > ret_best_acc:
            ret_best_acc = tp1_p2s_acc
            rgan_wrapper.save_networks(epoch, dirname(log_file), best_acc=ret_best_acc, is_best=True)
            log_print('>>Save best model: epoch={} best_acc(tp1_p2s):{:.2f}'.format(epoch + 1, tp1_p2s_acc))

        # Progam stastics
        rss, vms = get_sys_mem()
        log_print('Memory usage: rss={:.2f}GB vms={:.2f}GB Time:{:.2f}s'.format(rss, vms, time.time() - start_time))





