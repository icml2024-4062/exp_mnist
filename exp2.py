import torch
# from dynLABRBF import *
# from cla_LAB_ori import LAB_ori
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from dynMnist import *
# from dynLABfix import dyn_mnist_fix
# from data_reg import Yacht, Tecator, \
#      Comp_activ, Parkinson, SML, Airfoil, \
#      Tomshardware, Electricity, KCprice
# from fashionMnist import fashionMNIST_demo
# from dyn_cifar import cifar_demo
from data_MulCla import (MnistAll, Cifar5M, myFashionMnist, CifarStar,
                         myMnist, CifarMy, BreastCancer, myPCAM)
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Return a tuple (feature, label) for the given index
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])


def y_sv_cls(train_y, sv_num, cls=10):
    train_y_sign = train_y.argmax(dim=0)
    idx = np.empty((0))
    for label in range(cls):
        class_indices = np.where(train_y_sign == label)[0]
        idx_tmp = np.random.choice(class_indices, max(1, sv_num // cls), replace=False)
        idx = np.concatenate([idx, idx_tmp])
    if len(idx) > sv_num:
        idx = idx[:sv_num]
    return idx

def TrainKernel(LABRBF_model, train_x, train_y, optFlag=0, LR=1e-2, BS=64, isCls=False):

    criterion = nn.MSELoss(reduction='sum')

    # build the optimizer
    if optFlag < 1:
        optimizer = optim.Adam(LABRBF_model.parameters(), lr=LR)
        epochs = 1
    else:
        # optimizer = torch.optim.SGD(LABRBF_model.parameters(), lr=1e-4)
        optimizer = optim.Adam(LABRBF_model.parameters(), lr=1e-3)
        epochs = 3
    scheduler = lr_scheduler.StepLR(optimizer, 5, 0.5)
    optimizer.zero_grad()

    # train the Kernel
    loss_list = []

    train_dataset = CustomDataset(train_x.T, train_y.T)
    train_dataloader = DataLoader(train_dataset, batch_size=BS, shuffle=True)

    for epoch in range(epochs):
        for batch_idx, (batch_x, batch_y) in enumerate(train_dataloader):
            LABRBF_model.train()
            input = batch_x.squeeze().reshape(len(batch_y), -1).T
            input = preX(input)
            val_pred = LABRBF_model(x_train=input.to(LABRBF_model.device))
            # target = nn.functional.one_hot(batch_y, num_classes=LABRBF_model.cla).float()
            val_loss = criterion(val_pred, batch_y.T.to(LABRBF_model.device)) # or rsse_loss
            optimizer.zero_grad()
            val_loss.backward()

            torch.nn.utils.clip_grad_norm_(LABRBF_model.parameters(), max_norm=10, norm_type=2)
            optimizer.step()

            tmp = (torch.sqrt(val_loss)).detach().cpu()
            loss_list.append(tmp)

        scheduler.step()


def TrainLABRBF(dataset, max_sv=100, WEIGHT=1, LR=1e-2, BS=64, ini_sv=100, isCls=False, dataRate=1):

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    data_sv_x, data_sv_y = dataset.get_sv_data()
    data_train_x, data_train_y = dataset.get_train_data()
    data_test_x, data_test_y = dataset.get_test_data()

    data_sv_y, data_train_y, data_test_y = preY_one_hot(data_sv_y, data_train_y, data_test_y)

    data_num = int(data_train_y.shape[1]*dataRate)
    idx = np.random.choice(range(data_train_y.shape[1]), data_num, replace=False)

    train_x = data_train_x[:, idx].float()
    train_y = data_train_y[:, idx].float()
    data_test_x = data_test_x.float()
    data_test_y = data_test_y.float()

    sv_id = y_sv_cls(train_y, ini_sv)
    sv_x = train_x[:, sv_id]
    sv_y = train_y[:, sv_id]

    del data_train_y, data_train_x, data_sv_x, data_sv_y

    weight_dim = train_x.shape[0]
    weight_last = (torch.tensor(WEIGHT).float() / float(train_x.shape[0])
                   * torch.ones(weight_dim, sv_x.shape[1]))

    inner_iter = 0
    while inner_iter < 2:
        sv_num = sv_x.shape[1]
        print('sv: ', sv_num)
        add_num = sv_num - weight_last.shape[1]
        weight_mean = torch.mean(weight_last).cpu()
        weight_ini = torch.cat((weight_last.cpu(), weight_mean*torch.ones(weight_dim, add_num)), 1)
        LABRBF_model = LAB_unroll(x_sv=sv_x.to(device), y_sv=sv_y.to(device), weight_ini=weight_ini, cla=sv_y.shape[0])
        LABRBF_model.float().to(device)
        LABRBF_model.eval()

        start_time = time.time()
        TrainKernel(LABRBF_model, train_x, train_y,
                                  optFlag=inner_iter, LR=LR, BS=BS, isCls=isCls)
        run_time = time.time() - start_time
        print('training time:', format(run_time))
        weight_last = LABRBF_model.weight.data.detach().cpu()

        start_time = time.time()
        print('the train error rate loss: ')
        train_err, pred_y = testFun_cls(LABRBF_model, train_x, train_y)
        run_time = time.time() - start_time
        print('forward time:', format(run_time))

        tt = max_sv - sv_num
        k = min(max(20, int(sv_num / 10)), tt)
        if k <= 0:
            inner_iter = inner_iter + 1
            continue

        idx = add_new_sv_cls(train_y, pred_y, k, LABRBF_model.cla)

        sv_x = torch.cat((sv_x, train_x[:, idx.reshape(-1)]), 1)
        sv_y = torch.cat((sv_y, train_y[:, idx.reshape(-1)]), 1)

    print('the final train error rate loss: ')
    if isCls:
        train_err, pred_y = testFun_cls(LABRBF_model, train_x, train_y)
    else:
        train_err, pred_y = testFun_reg(LABRBF_model, train_x, train_y)

    print('the final test error rate loss: ')
    if isCls:
        test_err, pred_y = testFun_cls(LABRBF_model, data_test_x, data_test_y)
    else:
        test_err, pred_y = testFun_reg(LABRBF_model, data_test_x, data_test_y)

    return test_err, train_err


if __name__ == '__main__':
    for max_sv in [100, 200, 400, 600]:
        test_err_list = []
        train_err_list = []

        for dataRate in [1, 0.8, 0.5, 0.3]:
            test_err_list1 = []
            train_err_list1 = []

            for repeat in range(1):
                # parameter for LAB RBF
                ini_sv = max_sv // 3
                WEIGHT = 100 #50mnist #100fashion mnist #1 cifar #
                # parameter for SGD
                LR = 1e-2
                BS = 256

                global seed
                seed = repeat
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)

                # # --------------Classification------------------
                dataset = myFashionMnist(required_sv=ini_sv)#myMnist(required_sv=ini_sv)
                # #Electricity()#KCprice(required_sv=ini_sv) #Tecator(required_sv=ini_sv)
                # #Parkinson(required_sv=ini_sv) # KCprice() #Tecator()
                        # # Yacht() #
                        # # #Comp_activ() #SML() # Airfoil()
                        # # #
                test_err, train_err = TrainLABRBF(dataset, max_sv=max_sv, WEIGHT=WEIGHT, ini_sv=ini_sv,
                                                  LR=LR, BS=BS, isCls=True, dataRate=dataRate)
                test_err_list1.append(test_err)
                train_err_list1.append(train_err)

            test_err_list.append(torch.tensor(test_err_list1))
            train_err_list.append(torch.tensor(train_err_list1))

        torch.save(test_err_list, f'test_err_fashionmnist_{max_sv}.pt')
        torch.save(train_err_list, f'train_err_fashionmnist_{max_sv}.pt')

    # print('test err mean: ', torch.mean(test_err_list), 'err1 std: ',torch.std(test_err_list))

    # plt.show()
    print('finish')