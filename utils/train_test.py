import time
from datetime import datetime

from utils.quantizer import *

'''
    train_loader : 73257, #drop_last : 572 * 128, iter = 572
    test_loader : 26032, #drop_last : 203 * 128, iter = 203

    batch_idx : torch.Size([128])
    
    input : torch.Size([128, 3, 40, 40])
    target : torch.Size([128])
    
    output : torch.Size([128, 10])
    predicted : torch.Size([128])

'''


def train(epoch, train_loader, len_train_dataset, model, criterion, optimizer, gbit, tf_logger):
    print('\nEpoch: %d' % epoch)
    model.train()
    start_time = time.time()

    correct = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):

        # create hook handle
        model, handle_grad_list = net_grad_qn(model, gbit)

        # forward pass
        outputs = model(inputs.cuda())
        loss = criterion(outputs, targets.cuda())

        # TODO: training accuracy here
        _, predicted = torch.max(outputs.data, 1)
        correct += predicted.cpu().eq(targets.data).cpu().sum().item()

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # remove handle
        for handle_grad in handle_grad_list:
            handle_grad.remove()

        if batch_idx % 100 == 0:
            duration = time.time() - start_time

            print('%s epoch: %d step: %d cls_loss= %.5f (%d samples/sec)' %
                  (datetime.now(), epoch, batch_idx, loss.item(),
                   128 * 100 / duration))

    acc = 100. * correct / len_train_dataset
    tf_logger.add_scalar("train_acc", acc, epoch)

    print('%s------------------------------------------------------ '
          'Train Precision@1: %.2f%% \n' % (datetime.now(), acc))


def test(epoch, eval_loader, len_eval_dataset, model, tf_logger):
    model.eval()

    correct = 0

    for batch_idx, (inputs, targets) in enumerate(eval_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct += predicted.eq(targets.data).cpu().sum().item()

    # compute accuracy
    acc = 100. * correct / len_eval_dataset
    tf_logger.add_scalar("test_acc", acc, epoch)

    print('%s------------------------------------------------------ '
          'Test Precision@1: %.2f%% \n' % (datetime.now(), acc))
