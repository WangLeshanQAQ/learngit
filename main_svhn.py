import torch
from models.svhn_digit_dorefa import *
from utils.data_preprocess import *
from utils.train_test import *

# define model, Loss function, optimizer, lr_scheduler
wbit = 32
abit = 32
gbit = 32

train_loader, eval_loader, len_eval_dataset = svhn_dataset()

net = DigitNet_Q(wbit, abit).cuda()
criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)
# lr_schedu = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100, 150, 180], gamma=0.1)

# train the network
for epoch in range(300):
    train(epoch, train_loader, net, criterion, optimizer, gbit)
    test(eval_loader, net, len_eval_dataset)
    # lr_schedu.step(epoch)