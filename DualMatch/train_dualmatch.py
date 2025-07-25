import argparse
import math
import os
import random
import shutil
import time
import torch
import torch.nn
import torch.backends.cudnn as cudnn
import torch.utils.data
from resnet import *
from utils.utils_data import *
from utils.utils_loss import *
from utils.cub200 import load_cub200

from utils.voc import load_voc
from utils.stl10 import load_stl10

# 使用nips和kdd采样方式，训练不融合，测试融合
# 只使用concont loss
# 尾部分类器不使用LA后的输出指导模型学习
# 使用max全局阈值

torch.set_printoptions(precision=2, sci_mode=False)

parser = argparse.ArgumentParser(description='PyTorch implementation')
parser.add_argument('--dataset', default='cifar10', type=str,
                    choices=['cifar10', 'cifar100', 'sun397', 'voc', 'stl10'],
                    help='dataset name (cifar10)')
# parser.add_argument('--num-class', default=10, type=int,
#                     help='number of class')
parser.add_argument('--exp-dir', default='experiment_dualmatch', type=str,
                    help='experiment directory for saving checkpoints and logs')
parser.add_argument('--data_dir', default='../data', type=str,
                    help='experiment directory for loading pre-generated data')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=['resnet18'],
                    help='network architecture (only resnet18 supported)')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=800, type=int,
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-lr_decay_epochs', type=str, default='700,800',
                    help='where to decay lrt')
parser.add_argument('-lr_decay_rate', type=float, default=0.1,
                    help='decay rate for learning rate')
parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-3)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    help='print frequency (default: 100)')
parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--eta', default=0.9, type=float,
                    help='final weight of reliable sample loss')
parser.add_argument('--t', default=2, type=float,
                    help='tau for logits-adjustment')
parser.add_argument('--alpha_range', default='0.2,0.6', type=str,
                    help='ratio of clean labels (alpha)')
parser.add_argument('--e', default=50, type=int,
                    help='warm-up training')

parser.add_argument('--partial_rate', default=0.3, type=float,
                    help='ambiguity level')
parser.add_argument('--hierarchical', default=False, type=bool,
                    help='for CIFAR-100 fine-grained training')
parser.add_argument('--imb_type', default='exp', choices=['exp', 'step'],
                    help='imbalance data type')
parser.add_argument('--imb_ratio', default=100, type=float,
                    help='imbalance ratio for long-tailed dataset generation')
parser.add_argument('--save_ckpt', action='store_true',
                    help='whether save the model')

parser.add_argument('--resume', default='', type=str, help='models path for load')

parser.add_argument('--balanced', default=False, type=bool)
parser.add_argument('--only_pl', default='False', type=str)
parser.add_argument('--rho', default=0.2, type=float)


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Trainer():
    def __init__(self, args):
        self.args = args
        model_path = '{ds}_rho_{rho}_pr_{pr}_lt_{lt}_only_pl_{lb_flag}'.format(
            ds=args.dataset,
            rho=args.rho,
            pr=args.partial_rate,
            lt=int(args.imb_ratio),
            lb_flag=args.only_pl)
        args.exp_dir = os.path.join(args.exp_dir, str(args.seed), args.dataset, model_path)
        if not os.path.exists(args.exp_dir):
            os.makedirs(args.exp_dir)

        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        if args.dataset == 'cifar10':
            args.num_class = 10
            many_shot_num = 3
            low_shot_num = 3
            train_loader, train_givenY, test_loader, est_loader, init_label_dist, train_label_cnt \
                = load_cifar(args=args)

        elif args.dataset == 'cifar100':
            args.num_class = 100
            many_shot_num = 33
            low_shot_num = 33
            train_loader, train_givenY, test_loader, est_loader, init_label_dist, train_label_cnt \
                = load_cifar(args=args)

        elif args.dataset == 'stl10':
            args.batch_size = 2 * args.batch_size
            train_loader, train_givenY, test_loader, train_label_cnt = load_stl10(partial_rate=args.partial_rate,
                                                                                  batch_size=args.batch_size)
            args.num_class = 10
            many_shot_num = 3
            low_shot_num = 3
        # elif args.dataset == 'sun397':
        #     input_size = 224
        #     args.num_class = 397
        #     many_shot_num = 132
        #     low_shot_num = 132
        #     train_loader, train_givenY, test_loader, est_loader, init_label_dist, train_label_cnt = load_sun397(
        #         data_dir=args.data_dir,
        #         input_size=input_size,
        #         partial_rate=args.partial_rate,
        #         batch_size=args.batch_size)
        elif args.dataset == 'voc':
            train_loader, train_givenY, test_loader, train_label_cnt = load_voc(
                batch_size=args.batch_size, con=True)
            args.num_class = 20
            many_shot_num = 6
            low_shot_num = 7
        else:
            raise NotImplementedError("You have chosen an unsupported dataset. Please check and try again.")
        # this train loader is the partial label training loader

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_givenY = train_givenY.cuda()
        # set loss functions (with pseudo-targets maintained)
        self.acc_shot = AccurracyShot(train_label_cnt, args.num_class, many_shot_num, low_shot_num)

    def train(self, test_best=False):
        # create model
        print("=> creating model 'resnet18'")
        if args.dataset in ['sun397', 'voc']:
            print('Loading Pretrained Model')
            model = DHNet_Atten(args.num_class, pretrained=True)
        else:
            model = DHNet_Atten(args.num_class, standard=args.dataset == 'stl10')

        model = model.cuda()

        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=args.weight_decay)
        # set optimizer
        loss_fn = PLL_loss(self.train_givenY, mu=0.6)
        self.loss_fn = loss_fn

        args.start_epoch = 0

        if test_best:
            args.resume = os.path.join(args.exp_dir, 'checkpoint_best_ens.pth.tar')
        else:
            args.resume = os.path.join(args.exp_dir, 'checkpoint.pth.tar')

        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                self.loss_fn.confidence = checkpoint['confidence'].cuda()
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        if test_best:
            res, pred_list, true_list = self.test(model, self.test_loader)
            best_acc_ens, best_acc_many_ens, best_acc_med_ens, best_acc_few_ens = res[8:]
            with open(os.path.join(args.exp_dir, 'result.log'), 'a+') as f:
                f.write(f'Best res: \n')
                f.write('Best Acc {:.2f}, Shot - Many {:.2f}/ Med {:.2f}/Few {:.2f}. (lr {:.5f})\n'.format(
                        best_acc_ens, best_acc_many_ens, best_acc_med_ens, best_acc_few_ens,
                        optimizer.param_groups[0]['lr']))

            if args.dataset == 'cifar10':
                # 计算混淆矩阵并保存为pdf
                _, pred_list = torch.max(pred_list, dim=1)
                confusion_matrix_test = torch.zeros((args.num_class, args.num_class)).cuda()
                label_all = true_list.long()
                pred_all = pred_list.long()
                for i in range(len(label_all)):
                    confusion_matrix_test[label_all[i], pred_all[i]] += 1
                confusion_matrix_test = confusion_matrix_test / (confusion_matrix_test.sum(dim=1) + 1e-10).repeat(
                    confusion_matrix_test.size(1), 1).transpose(0, 1)

                # 转换为 NumPy 并先保留两位小数
                confusion_matrix = confusion_matrix_test.cpu().numpy()
                confusion_matrix = np.round(confusion_matrix, 2)

                # 按行归一化（防止除以 0 的问题）
                row_sums = confusion_matrix.sum(axis=1, keepdims=True)
                confusion_matrix = np.divide(confusion_matrix, row_sums, out=np.zeros_like(confusion_matrix),
                                             where=row_sums != 0)
                confusion_matrix = np.round(confusion_matrix, 2)  # 再次确保保留两位小数

                # 类别标签
                class_labels = [str(i) for i in range(confusion_matrix.shape[0])]

                # 设置图片清晰度
                plt.rcParams['figure.dpi'] = 600
                plt.rcParams['font.size'] = 18  # 全局字体大小调整（比如 14，可再加大）

                # 创建画布
                plt.figure(figsize=(10, 8))

                # 绘制混淆矩阵
                plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
                # plt.title('Confusion matrix')
                plt.colorbar()

                # 设置 x 轴和 y 轴的刻度标签
                tick_marks = np.arange(len(class_labels))
                plt.xticks(tick_marks, class_labels)
                plt.yticks(tick_marks, class_labels)

                # 添加数值标签
                thresh = confusion_matrix.max() / 2.
                for i in range(confusion_matrix.shape[0]):
                    for j in range(confusion_matrix.shape[1]):
                        if round(confusion_matrix[i, j], 2) == 0:
                            plt.text(j, i, format(confusion_matrix[i, j], '.0f'),
                                     horizontalalignment="center",
                                     color="white" if confusion_matrix[i, j] > thresh else "black")
                        else:
                            plt.text(j, i, format(confusion_matrix[i, j], '.2f'),
                                     horizontalalignment="center",
                                     color="white" if confusion_matrix[i, j] > thresh else "black")

                plt.tight_layout()
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')

                # 保存为 PDF 文件
                plt.savefig(args.exp_dir + '/dualmatch_confusion_matrix.pdf', format='pdf', bbox_inches='tight',
                            pad_inches=0)

                # 显示图形
                plt.show()
            return

        best_acc_head = 0
        best_acc_tail = 0
        best_acc_ens = 0

        for epoch in range(args.start_epoch, args.epochs):
            is_best_ens = False

            adjust_learning_rate(args, optimizer, epoch)

            self.train_loop(model, loss_fn, optimizer, epoch)

            res, _, _ = self.test(model, self.test_loader)
            acc_test_head, acc_many_head, acc_med_head, acc_few_head = res[:4]
            acc_test_tail, acc_many_tail, acc_med_tail, acc_few_tail = res[4:8]
            acc_test_ens, acc_many_ens, acc_med_ens, acc_few_ens = res[8:]

            time_now = time.strftime("%Y-%m-%d %H:%M:%S")

            with open(os.path.join(args.exp_dir, 'result.log'), 'a+') as f:
                f.write(f'time: {time_now} \n')
                f.write(
                    'Epoch {}/{}: Acc_head {:.2f}, Best Acc_head {:.2f}, Shot - Many {:.2f}/ Med {:.2f}/Few {:.2f}. (lr {:.5f})\n'.format(
                        epoch, args.epochs, acc_test_head, best_acc_head, acc_many_head, acc_med_head, acc_few_head,
                        optimizer.param_groups[0]['lr']))
                f.write(
                    'Epoch {}/{}: Acc_tail {:.2f}, Best Acc_tail {:.2f}, Shot - Many {:.2f}/ Med {:.2f}/Few {:.2f}. (lr {:.5f})\n'.format(
                        epoch, args.epochs, acc_test_tail, best_acc_tail, acc_many_tail, acc_med_tail, acc_few_tail,
                        optimizer.param_groups[0]['lr']))
                f.write(
                    'Epoch {}/{}: Acc_ens {:.2f}, Best Acc_ens {:.2f}, Shot - Many {:.2f}/ Med {:.2f}/Few {:.2f}. (lr {:.5f})\n'.format(
                        epoch, args.epochs, acc_test_ens, best_acc_ens, acc_many_ens, acc_med_ens, acc_few_ens,
                        optimizer.param_groups[0]['lr']))

            if acc_test_head > best_acc_head:
                best_acc_head = acc_test_head

            if acc_test_tail > best_acc_tail:
                best_acc_tail = acc_test_tail

            if acc_test_ens > best_acc_ens:
                best_acc_ens = acc_test_ens
                is_best_ens = True

            if args.save_ckpt:
                self.save_checkpoint({
                    'confidence': loss_fn.confidence.detach(),
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best=is_best_ens, filename='{}/checkpoint.pth.tar'.format(args.exp_dir),
                    best_file_name='{}/checkpoint_best_ens.pth.tar'.format(args.exp_dir))
            # save checkpoint

    def get_high_confidence(self, loss_vec, pseudo_label_idx, nums_vec):
        idx_chosen = []
        chosen_flags = torch.zeros(len(loss_vec)).cuda()
        # initialize selection flags
        for j, nums in enumerate(nums_vec):
            indices = np.where(pseudo_label_idx.cpu().numpy() == j)[0]
            # torch.where will cause device error
            if len(indices) == 0:
                continue
                # if no sample is assigned this label1 (by argmax), skip
            loss_vec_j = loss_vec[indices]
            sorted_idx_j = loss_vec_j.sort()[1].cpu().numpy()
            partition_j = max(min(int(math.ceil(nums)), len(indices)), 1)
            # at least one example
            idx_chosen.append(indices[sorted_idx_j[:partition_j]])

        idx_chosen = np.concatenate(idx_chosen)
        chosen_flags[idx_chosen] = 1

        idx_chosen = torch.where(chosen_flags == 1)[0]
        return idx_chosen

    def get_loss(self, X_w, logits_w, logits_s, ce_label, Y, index, model, loss_fn, emp_dist, alpha, eta, epoch,
                 is_tail):
        # 计算HTC论文公式4和5
        if epoch > 800:
            loss_pseu, _ = loss_fn(logits_w, index)
        else:
            pred_gradient = F.softmax(logits_w, dim=1)
            num_label = Y.sum(dim=1)  # [B]
            num_class = Y.size(1)
            valid_mask = (num_label <= num_class - 1)  # [B]
            if valid_mask.any():
                pred_valid = pred_gradient[valid_mask]  # [N, C]
                Y_valid = Y[valid_mask]  # [N, C]
                probs = torch.sum(pred_valid * Y_valid, dim=1)  # [N]
                loss_pseu = -torch.log(probs + 1e-10).mean()
            else:
                loss_pseu = torch.tensor(0.0).cuda()

        if is_tail:
            # tail
            with (torch.no_grad()):
                pseudo_score = Y * F.softmax(logits_w.detach() - args.t * torch.log(emp_dist), dim=1) + 1e-10
                pseudo_score = pseudo_score / pseudo_score.sum(dim=1, keepdim=True)
                tau_tail = 0
                for label_idx in range(pseudo_score.size(1)):
                    idx_high = pseudo_score[:, label_idx].sort(descending=True)[1][
                               :int(len(pseudo_score) / args.num_class)]
                    tau_tail += pseudo_score[idx_high, label_idx].sum().item()
                selected_num = min(int(tau_tail / pseudo_score.size(1)), int(len(pseudo_score) / pseudo_score.size(1)))
                selected_num = max(selected_num, 1)

                pseudo_idx_high = torch.tensor([], dtype=torch.int64).cuda()
                pseudo_label_high = torch.tensor([], dtype=torch.int64).cuda()
                for label_idx in range(pseudo_score.size(1)):
                    per_label_mask = pseudo_score[:, label_idx].sort(descending=True)[1][:selected_num]
                    pseudo_idx_high = torch.cat((pseudo_idx_high, per_label_mask))
                    pseudo_label_high = torch.cat(
                        (pseudo_label_high, torch.tensor([label_idx] * selected_num).cuda()))
        else:
            # head
            with torch.no_grad():
                # head按照nips方法选高置信样本
                prediction = F.softmax(logits_w.detach(), dim=1)
                prediction_adj = prediction * Y
                prediction_adj = prediction_adj / prediction_adj.sum(dim=1, keepdim=True)
                pseudo_score = prediction_adj

                max_values, _ = torch.max(pseudo_score, dim=0)
                max_values_mean = max_values.mean()

                selected_num = torch.tensor([0.] * pseudo_score.size(1)).cuda()
                num_each_calss = torch.round(emp_dist * len(pseudo_score)).to(int)
                for label_idx in range(pseudo_score.size(1)):
                    sort_class = pseudo_score[:, label_idx].cpu().sort(descending=True)
                    sort_probs = sort_class[0]
                    selected_num[label_idx] = sum(sort_probs[:num_each_calss[label_idx]])
                selected_num = torch.max(
                    torch.round(selected_num * max_values_mean).to(int),
                    torch.tensor(1))

                pseudo_idx_high = torch.tensor([], dtype=torch.int64).cuda()
                pseudo_label_high = torch.tensor([], dtype=torch.int64).cuda()
                for label_idx in range(pseudo_score.size(1)):
                    per_label_mask = logits_w[:, label_idx].sort(descending=True)[1][
                                     :selected_num[label_idx]]
                    pseudo_idx_high = torch.cat((pseudo_idx_high, per_label_mask))
                    pseudo_label_high = torch.cat(
                        (pseudo_label_high, torch.tensor([label_idx] * selected_num[label_idx]).cuda().to(torch.int64)))

        if epoch < 1 or pseudo_idx_high.shape[0] == 0:
            # first epoch, using uniform labels for training
            # if no samples are chosen
            loss = loss_pseu
        else:
            # consistency regularization
            loss_ce = (F.cross_entropy(logits_s[pseudo_idx_high, :], pseudo_label_high, reduction='none')).mean()

            pseudo_label_high_one_hot = F.one_hot(pseudo_label_high, num_classes=args.num_class)
            l = np.random.beta(4, 4)
            l = max(l, 1 - l)
            X_w_c = X_w[pseudo_idx_high]
            ce_label_c = pseudo_label_high_one_hot
            idx = torch.randperm(X_w_c.size(0))
            X_w_c_rand = X_w_c[idx]
            ce_label_c_rand = ce_label_c[idx]
            X_w_c_mix = l * X_w_c + (1 - l) * X_w_c_rand
            ce_label_c_mix = l * ce_label_c + (1 - l) * ce_label_c_rand
            if is_tail:
                _, logits_mix, _ = model(X_w_c_mix)
            else:
                logits_mix, _, _ = model(X_w_c_mix)
            loss_mix, _ = loss_fn(logits_mix, None, targets=ce_label_c_mix)
            # mixup training

            loss = (loss_ce + loss_mix) * eta + loss_pseu

        return loss, pseudo_score

    def train_loop(self, model, loss_fn, optimizer, epoch):
        args = self.args
        train_loader = self.train_loader

        batch_time = AverageMeter('Time', ':1.2f')
        data_time = AverageMeter('DataTime', ':1.2f')
        acc_head = AverageMeter('Acc@head', ':2.2f')
        acc_con = AverageMeter('Acc@con', ':2.2f')
        acc_tail = AverageMeter('Acc@tail', ':2.2f')
        acc_en = AverageMeter('Acc@en', ':2.2f')
        loss_head_log = AverageMeter('Loss@head', ':2.2f')
        loss_tail_log = AverageMeter('Loss@tail', ':2.2f')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, acc_head, acc_con, acc_tail, acc_en, loss_head_log, loss_tail_log],
            prefix="Epoch: [{}]".format(epoch))

        # switch to train mode
        model.train()

        eta = args.eta * linear_rampup(epoch, args.e)
        alpha = args.alpha_start + (args.alpha_end - args.alpha_start) * linear_rampup(epoch, args.e)
        # calculate weighting parameters

        end = time.time()

        emp_dist_tail = torch.Tensor([1 / args.num_class for _ in range(args.num_class)]).cuda()

        emp_dist_head = loss_fn.confidence.sum(0) / loss_fn.confidence.sum()

        for i, (images_w, images_s, labels, true_labels, index) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            # x_s is just used for adding more samples
            X_w, X_s, Y, index = images_w.cuda(), images_s.cuda(), labels.cuda(), index.cuda()
            Y_true = true_labels.long().detach().cuda()
            # for showing training accuracy and will not be used when training

            logits_w_head, logits_w_tail, feat_w = model(X_w)
            logits_s_head, logits_s_tail, feat_s = model(X_s)
            pseudo_label = loss_fn.confidence[index]

            loss_head, prediction_head = self.get_loss(X_w, logits_w_head, logits_s_head, pseudo_label, Y, index, model,
                                                       loss_fn, emp_dist_head, alpha, eta, epoch, False)

            logit_adj = F.softmax(logits_w_head - args.t * torch.log(emp_dist_head), dim=1)
            loss_tail, prediction_tail = self.get_loss(X_w, logits_w_tail, logits_s_tail, logit_adj, Y, index, model,
                                                       loss_fn, emp_dist_head, alpha, eta, epoch, True)

            loss = loss_head + loss_tail

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_head_log.update(loss_head.item())
            loss_tail_log.update(loss_tail.item())
            # log accuracy
            acc = accuracy(logits_w_head, Y_true)[0]
            acc_head.update(acc[0])

            acc = accuracy(logits_w_tail, Y_true)[0]
            acc_tail.update(acc[0])

            # acc = accuracy(fusion_pred.detach(), Y_true)[0]
            # acc_en.update(acc[0])

            acc = accuracy(pseudo_label, Y_true)[0]
            acc_con.update(acc[0])

            loss_fn.confidence_move_update(prediction_tail, index)
            # update confidences

            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                progress.display(i)

    def test(self, model, test_loader):
        def cal(acc_shot, pred_list, true_list):
            acc1, acc5 = accuracy(pred_list, true_list, topk=(1, 5))
            acc_many, acc_med, acc_few = acc_shot.get_shot_acc(pred_list.max(dim=1)[1], true_list)
            print('==> Test Accuracy is %.2f%% (%.2f%%), [%.2f%%, %.2f%%, %.2f%%]' % (
                acc1, acc5, acc_many, acc_med, acc_few))
            return float(acc1), float(acc_many), float(acc_med), float(acc_few)

        with torch.no_grad():
            model.eval()
            imb_pred_list = []
            bal_pred_list = []
            fusion_pred_list = []
            true_list = []
            for _, (images, labels) in enumerate(test_loader):
                images = images.cuda()
                imb_output, bal_output, _ = model(images)
                fusion_pred = model.ensemble(imb_output, bal_output, self.loss_fn.get_distribution())
                bal_pred = F.softmax(bal_output, dim=1)
                imb_pred = F.softmax(imb_output, dim=1)

                imb_pred_list.append(imb_pred.cpu())
                bal_pred_list.append(bal_pred.cpu())
                fusion_pred_list.append(fusion_pred.cpu())
                true_list.append(labels)

            imb_pred_list = torch.cat(imb_pred_list, dim=0)

            bal_pred_list = torch.cat(bal_pred_list, dim=0)

            fusion_pred_list = torch.cat(fusion_pred_list, dim=0)
            true_list = torch.cat(true_list, dim=0)

        res = []
        print('==> Evaluation imb...')
        res.extend(list(cal(self.acc_shot, imb_pred_list, true_list)))
        print('==> Evaluation bal...')
        res.extend(list(cal(self.acc_shot, bal_pred_list, true_list)))
        print('==> Evaluation ensemble...')
        res.extend(list(cal(self.acc_shot, fusion_pred_list, true_list)))

        return res, fusion_pred_list, true_list

    # def test(self, model, test_loader, type=1):
    #     with torch.no_grad():
    #         if type == 1:
    #             print('==> Evaluation tail...')
    #         elif type == 2:
    #             print('==> Evaluation head...')
    #         else:
    #             print('==> Evaluation ensemble...')
    #         model.eval()
    #         pred_list = []
    #         true_list = []
    #         for _, (images, labels) in enumerate(test_loader):
    #             images = images.cuda()
    #             if type == 1:
    #                 _, outputs, _ = model(images)
    #                 pred = F.softmax(outputs, dim=1)
    #             elif type == 2:
    #                 outputs, _, _ = model(images)
    #                 pred = F.softmax(outputs, dim=1)
    #             else:
    #                 # 使用熵
    #                 logit_head, logit_tail, _ = model(images)
    #                 pred_head = torch.softmax(logit_head.detach(), dim=-1)
    #                 pred_tail = torch.softmax(logit_tail.detach(), dim=-1)
    #                 w1, w2 = ensemble_fusion(pred_head, pred_tail)
    #                 pred = w1 * pred_head + w2 * pred_tail
    #
    #             pred_list.append(pred.cpu())
    #             true_list.append(labels)
    #
    #         pred_list = torch.cat(pred_list, dim=0)
    #         true_list = torch.cat(true_list, dim=0)
    #
    #         acc1, acc5 = accuracy(pred_list, true_list, topk=(1, 5))
    #         acc_many, acc_med, acc_few = self.acc_shot.get_shot_acc(pred_list.max(dim=1)[1], true_list)
    #         print('==> Test Accuracy is %.2f%% (%.2f%%), [%.2f%%, %.2f%%, %.2f%%]' % (
    #             acc1, acc5, acc_many, acc_med, acc_few))
    #     return float(acc1), float(acc_many), float(acc_med), float(acc_few), pred_list, true_list

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar', best_file_name='model_best.pth.tar'):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, best_file_name)


if __name__ == '__main__':
    args = parser.parse_args()

    [args.alpha_start, args.alpha_end] = [float(item) for item in args.alpha_range.split(',')]
    iterations = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([])
    for it in iterations:
        args.lr_decay_epochs.append(int(it))

    args.imb_factor = 1. / args.imb_ratio
    print(args)

    # set imb_factor as 1/imb_ratio
    trainer = Trainer(args)
    trainer.train()
    trainer.train(test_best=True)
