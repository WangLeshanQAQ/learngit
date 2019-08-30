import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from models.svhn_digit_dorefa import *
from data.data_preprocess import *
from utils.train_test import *

wbit = 1
abit = 1
gbit = 2


def main():
    # load dataset
    train_loader, eval_loader, len_train_dataset, len_eval_dataset = svhn_dataset()

    # define model, Loss function, optimizer, lr_scheduler
    model = DigitNet_Q(wbit, abit).cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    tf_logger = SummaryWriter("./tf_logs")

    for epoch in range(200):
        train(epoch, train_loader, len_train_dataset, model, criterion, optimizer, gbit, tf_logger)
        test(epoch, eval_loader, len_eval_dataset, model, tf_logger)
    
    tf_logger.close()


if __name__ == "__main__":
    # if mp.get_start_method(allow_none=True) != "forkserver":
    #    mp.set_start_method("forkserver")

    main()
