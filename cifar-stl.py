import os
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim

from models import LeNet
from utils import *

# PATHS
CHECKPOINT = "./checkpoints/cifar-stl"

# BATCH
BATCH_SIZE = 256
NUM_WORKERS = 4

# SGD
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

# Step Decay
LR_DROP = 0.5
EPOCHS_DROP = 20

# MISC
EPOCHS = 100
CUDA = True

# Manual seed
SEED = 20

random.seed(SEED)
torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed_all(SEED)

ALL_CLASSES = range(10)


def main():
    if not os.path.isdir(CHECKPOINT):
        os.makedirs(CHECKPOINT)

    print('==> Preparing dataset')

    trainloader, testloader = load_CIFAR(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    CLASSES = []
    AUROCs = []
    auroc = AverageMeter()

    for t, cls in enumerate(ALL_CLASSES):

        print('\nTask: [%d | %d]\n' % (t + 1, len(ALL_CLASSES)))

        CLASSES = [cls]

        print("==> Creating model")
        model = LeNet(num_classes=1)

        if CUDA:
            model = model.cuda()
            model = nn.DataParallel(model)
            cudnn.benchmark = True

        print('    Total params: %.2fK' % (sum(p.numel() for p in model.parameters()) / 1000))

        criterion = nn.BCELoss()
        optimizer = optim.SGD(model.parameters(),
                              lr=LEARNING_RATE,
                              momentum=MOMENTUM,
                              weight_decay=WEIGHT_DECAY
                              )

        print("==> Learning")

        best_loss = 1e10
        learning_rate = LEARNING_RATE

        for epoch in range(EPOCHS):

            # decay learning rate
            if (epoch + 1) % EPOCHS_DROP == 0:
                learning_rate *= LR_DROP
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate

            print('Epoch: [%d | %d]' % (epoch + 1, EPOCHS))

            train_loss = train(trainloader, model, criterion, CLASSES, CLASSES, optimizer=optimizer, use_cuda=CUDA)
            test_loss = train(testloader, model, criterion, CLASSES, CLASSES, test=True, use_cuda=CUDA)

            # save model
            is_best = test_loss < best_loss
            best_loss = min(test_loss, best_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'loss': test_loss,
                'optimizer': optimizer.state_dict()
            }, CHECKPOINT, is_best)

        print("==> Calculating AUROC")

        filepath_best = os.path.join(CHECKPOINT, "best.pt")
        checkpoint = torch.load(filepath_best)
        model.load_state_dict(checkpoint['state_dict'])

        new_auroc = calc_avg_AUROC(model, testloader, CLASSES, CLASSES, CUDA)
        auroc.update(new_auroc)

        print('New Task AUROC: {}'.format(new_auroc))
        print('Average AUROC: {}'.format(auroc.avg))

        AUROCs.append(auroc.avg)

    print('\nAverage Per-task Performance over number of tasks')
    for i, p in enumerate(AUROCs):
        print("%d: %f" % (i + 1, p))


if __name__ == '__main__':
    main()
