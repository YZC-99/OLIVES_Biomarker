import torch
from utils.utils import AverageMeter,warmup_learning_rate
import sys
import time
import numpy as np
from sklearn.metrics import roc_auc_score
from ex_config.config_linear import parse_option
from utils.utils import set_loader_new, set_model, set_optimizer, adjust_learning_rate
from utils.utils import save_model
import os

def main_bce_unlock_layer4():
    best_acc = 0
    opt = parse_option()
    current_dir = opt.results_dir
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
    # build data loader
    device = opt.device
    train_loader,  test_loader = set_loader_new(opt)


    model, classifier, criterion = set_model(opt)

    optimizer = set_optimizer(opt, classifier)
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc = train_OCT_unlock_layer4(train_loader, model, classifier, criterion,
                          optimizer, epoch, opt)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, acc))
        # loss, test_acc,r = validate(test_loader, model, classifier, criterion, opt)

        # if test_acc > best_acc:
        #     best_acc = test_acc
        #     print('Best accuracy found at epoch {}, accuracy: {:.2f}'.format(epoch, best_acc))
        #     torch.save(model, os.path.join(current_dir, 'best_model.pth'))
        if epoch % 5 == 0:
            state = {
                'model': model.state_dict(),
                'epoch': epoch,
            }
            torch.save(state, os.path.join(current_dir, '{}_model.pth'.format(epoch)))
            del state



def train_OCT_unlock_layer4(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    device = opt.device
    end = time.time()
    for idx, all_info in enumerate(train_loader):
        data_time.update(time.time() - end)
        image = all_info['image']
        images = image.to(device)

        if (opt.biomarker == 'vit_deb'):
            labels = all_info['vit_deb']
        elif (opt.biomarker == 'ir_hrf'):
            labels = all_info['ir_hrf']
        elif (opt.biomarker == 'full_vit'):
            labels = all_info['full_vit']
        elif (opt.biomarker == 'partial_vit'):
            labels = all_info['partial_vit']
        elif (opt.biomarker == 'drt'):
            labels = all_info['drt']
        elif (opt.biomarker == 'fluid_srf'):
            labels = all_info['fluid_srf']
        else:
            labels = all_info['fluid_irf']
        labels = labels.long()
        bsz = labels.shape[0]
        labels=labels.to(device)
        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # model.encoder中，除了layer4的参数，其他参数都冻结
        for name, param in model.encoder.named_parameters():
            if 'layer4' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        features = model.encoder(images)
        # with torch.no_grad():
        #     features = model.encoder(images)

        output = classifier(features.detach())
        output=output.squeeze()
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()
    device = opt.device
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    label_list = []
    out_list = []
    with torch.no_grad():
        end = time.time()
        for idx, (image, vit_deb,ir_hrf, full_vit,partial_vit,fluid_irf,drt,eye_id,bcva,cst,patient) in enumerate(val_loader):
            images = image.float().to(device)

            if (opt.biomarker == 'vit_deb'):
                labels = vit_deb
            elif (opt.biomarker == 'ir_hrf'):
                labels = ir_hrf
            elif (opt.biomarker == 'full_vit'):
                labels = full_vit
            elif (opt.biomarker == 'partial_vit'):
                labels = partial_vit
            elif (opt.biomarker == 'drt'):
                labels = drt
            else:
                labels = fluid_irf
            labels = labels.long()

            label_list.append(labels.detach().cpu().numpy())
            labels = labels.to(device)
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))


            out_list.append(output.squeeze().detach().cpu().numpy())
            # update metric

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))
    # label_array = np.array(label_list)
    # out_array = np.array(out_list)
    # r = roc_auc_score(label_array, out_array, multi_class='ovr', average='weighted')
    # return losses.avg, top1.avg,r
    return losses.avg, top1.avg,0
