import numpy as np
import torch
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import get_sup_dataset
from model import ResUNet
from utils import create_dir, parse_arg
from utils import dice_loss, compute_metric

create_dir()


def train_supervised(args):
    # prepare train and validation dataset
    train_set, val_set = get_sup_dataset(args.data_path, args.train_val_ratio, args.labeled_ratio)

    # prepare dataloader
    train_loader = DataLoader(train_set, args.batch_size, True, num_workers=args.num_worker)
    val_loader = DataLoader(val_set, args.batch_size, True, num_workers=args.num_worker)

    # initialize network
    net = ResUNet().to(args.device)

    # define loss and optimizer
    criterion_dice = dice_loss
    criterion_ce = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    logging.info('start training!')
    best_dice = 0

    for epoch in range(args.epoch):

        # ####################################### train model #######################################

        loss_history = []

        # for data, mask in train_loader:
        for data, mask in tqdm(train_loader, desc='training progress', leave=False):
            data, mask = data.to(args.device), mask.to(args.device)

            # network predict
            out = net.forward(data)

            # compute loss
            loss = criterion_dice(out, mask) + criterion_ce(out, mask)

            # backward propagation and parameter update
            optim.zero_grad()
            loss.backward()
            optim.step()

            loss_history.append(loss.cpu().data.numpy())

        logging.info('epoch: %d/%d | train | dice loss: %.4f' % (epoch, args.epoch, float(np.mean(loss_history))))

        # ####################################### validate model #######################################

        # validation performance metrics
        pa = pa_total = 0
        iou = iou_total = 0
        dice = dice_total = 0

        with torch.no_grad():
            for data, mask in tqdm(val_loader, desc='validation progress', leave=False):
                data, mask = data.to(args.device), mask.to(args.device)

                # network predict
                out = net(data)
                out = torch.argmax(out, dim=1)

                # compute metrics
                result = compute_metric(out, mask)
                pa += result[0]
                iou += result[1]
                dice += result[2]
                pa_total += len(mask)
                iou_total += len(mask)
                dice_total += len(mask)

        logging.info('epoch: %d/%d |  val  | DICE: %.4f | PA: %.4f | IOU: %.4f' % (
            epoch, args.epoch, dice / dice_total, pa / pa_total, iou / iou_total))

        if dice > best_dice:
            best_dice = dice
            torch.save(net.state_dict(), './model/net_sup.pth')
            logging.info('best model | epoch: %d | DICE: %.4f | PA: %.4f | IOU: %.4f' % (
                epoch, dice / dice_total, pa / pa_total, iou / iou_total))


if __name__ == '__main__':
    logging.basicConfig(filename="log/train_sup.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')

    args = parse_arg()
    logging.info(args)

    train_supervised(args)
