import torch
from models.Alexnet import *
from utils.data_preprocess import *
from utils.train_test import *

# load data
train_loader, eval_loader, len_eval_dataset = cifar_dataset()

# define model, Loss function, optimizer, lr_scheduler
wbit = 32
abit = 32
gbit = 32

net = AlexNet_Q(wbit, abit).cuda()
criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
lr_schedu = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 150, 180], gamma=0.1)

# train the network
for epoch in range(200):
    train(epoch, train_loader, net, criterion, optimizer, gbit)
    test(eval_loader, net, len_eval_dataset)
    lr_schedu.step(epoch)

