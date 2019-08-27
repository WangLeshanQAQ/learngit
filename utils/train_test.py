import time
import torch
from datetime import datetime
from utils.quantizer import *


def train(epoch, train_loader, model, criterion, optimizer, gbit):
    print('\nEpoch: %d' % epoch)
    model.train()

    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):

        # create hanlde
        model, handle_grad_list = net_grad_qn(model, gbit)
        outputs = model(inputs.cuda())
        loss = criterion(outputs, targets.cuda())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # remove handle
        for handle_grad in handle_grad_list:
            handle_grad.remove()

        if batch_idx % 10 == 0:
            duration = time.time() - start_time

            print('%s epoch: %d step: %d cls_loss= %.5f (%d samples/sec)' %
                  (datetime.now(), epoch, batch_idx, loss.item(),
                   128 * 10 / duration))


def test(eval_loader, model, len_eval_dataset):
    # pass
    model.eval()
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(eval_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct += predicted.eq(targets.data).cpu().sum().item()

    acc = 100. * correct / len_eval_dataset
    print('%s------------------------------------------------------ '
          'Precision@1: %.2f%% \n' % (datetime.now(), acc))
