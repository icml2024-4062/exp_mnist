import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from cls_LAB_unroll import LAB_unroll
import matplotlib.pyplot as plt
import time
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST


class AddGaussianNoise(object):
    def __init__(self, mean=0, std=0.3):
        self.mean = mean
        self.std = std

    def __call__(self, img_tensor):

        # Generate Gaussian noise
        noise = torch.randn_like(img_tensor) * self.std + self.mean

        # Add noise to the image
        noisy_img = img_tensor + noise

        # Clip the pixel values to be in the valid range [0, 1]
        noisy_img = torch.clamp(noisy_img, 0, 1)

        return noisy_img


def testFun_reg(LABRBF_model, data_x, data_y):
    #
    batch = 128
    LABRBF_model.eval()
    pred_y_list = []
    cnt = 0
    while cnt < data_x.shape[1]:
        if cnt + batch < data_x.shape[1]:
            pred_y_list.append(LABRBF_model(x_train=data_x[:, cnt: cnt + batch]
                                            .to(LABRBF_model.device)).detach().cpu())
        else:
            pred_y_list.append(LABRBF_model(x_train=data_x[:, cnt:].to(LABRBF_model.device)).detach().cpu())
        cnt = cnt + batch

    pred_y = torch.cat(pred_y_list, 1)
    #
    pred_y = torch.clamp(pred_y, -1, 1)
    acc = 1 - LABRBF_model.rsse_loss(pred=pred_y, target=data_y).detach().cpu()

    print('\t\t the MAE loss: ', format(LABRBF_model.mae_loss(pred=pred_y, target=data_y)))
    print('\t\t the R2 loss: ', format(acc))
    print('\t\t the rmse loss:', format(LABRBF_model.rmse_loss(pred=pred_y, target=data_y)))

    return acc, pred_y


def testFun_cls(LABRBF_model, data_x, data_y):
    #
    batch = 128
    LABRBF_model.eval()
    pred_y_list = []
    criterion = nn.MSELoss(reduction='sum')

    cnt = 0
    while cnt < data_x.shape[1]:
        if cnt + batch < data_x.shape[1]:
            pred_y_list.append(LABRBF_model(x_train=data_x[:, cnt: cnt + batch].to(LABRBF_model.device)).detach().cpu())
        else:
            pred_y_list.append(LABRBF_model(x_train=data_x[:, cnt:].to(LABRBF_model.device)).detach().cpu())
        cnt = cnt + batch

    pred_y = torch.cat(pred_y_list, 1)
    acc = LABRBF_model.cls_loss(pred=pred_y, target=data_y.argmax(dim=0)).detach().cpu()
    print('\t\t the MSE loss: ', format(criterion(pred_y, data_y)))
    print('\t\t the Classification Acc: ', format(acc))

    return acc, pred_y

def TrainKernel(LABRBF_model, optFlag=0, LR=1e-2, BS=64, isCls=False):

    criterion = nn.MSELoss(reduction='sum')

    # build the optimizer
    if optFlag < 1:
        optimizer = optim.Adam(LABRBF_model.parameters(), lr=LR)
        epochs = 1
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor([0.1307]), std=torch.tensor([0.3081]))
            ])
    else:
        # optimizer = torch.optim.SGD(LABRBF_model.parameters(), lr=1e-4)
        optimizer = optim.Adam(LABRBF_model.parameters(), lr=1e-3)
        epochs = 3
        transform = transforms.Compose([
            # transforms.RandomRotation(degrees=5),
            # transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            # transforms.RandomResizedCrop(size=(28, 28), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor([0.1307]), std=torch.tensor([0.3081]))
            ])
    scheduler = lr_scheduler.StepLR(optimizer, 2, 0.5)
    optimizer.zero_grad()

    # train the Kernel
    loss_list = []

    mnist_train_dataset = MNIST(root='/users/sista/fhe/no_backup/dataset',
                                train=True, download=False, transform=transform)
    train_dataloader = DataLoader(mnist_train_dataset, batch_size=BS, shuffle=True)

    for epoch in range(epochs):
        for batch_idx, (batch_x, batch_y) in enumerate(train_dataloader):
            LABRBF_model.train()
            input = batch_x.squeeze().reshape(len(batch_y), -1).T
            input = preX(input)
            val_pred = LABRBF_model(x_train=input.to(LABRBF_model.device))
            target = nn.functional.one_hot(batch_y, num_classes=LABRBF_model.cla).float()
            val_loss = criterion(val_pred, target.T.to(LABRBF_model.device)) # or rsse_loss
            optimizer.zero_grad()
            val_loss.backward()

            torch.nn.utils.clip_grad_norm_(LABRBF_model.parameters(), max_norm=10, norm_type=2)
            optimizer.step()

            tmp = (torch.sqrt(val_loss)).detach().cpu()
            loss_list.append(tmp)

        scheduler.step()

        print('[{}] loss={}'.format(batch_idx, val_loss))
        for name, params in LABRBF_model.named_parameters():
            print('-->name:', name, ' -->grad_value:',  params.grad.data.norm(), '-->weight_value:', params.data.norm())

    plt.plot(loss_list[10:])

def preY_one_hot(sv_y, train_y, test_y):
    num_cls = train_y.max() + 1
    sv_y = nn.functional.one_hot(sv_y, num_classes=num_cls)
    train_y = nn.functional.one_hot(train_y, num_classes=num_cls)
    test_y = nn.functional.one_hot(test_y, num_classes=num_cls)

    return sv_y.T, train_y.T, test_y.T


def preX(data_x):
    # loaded_data = torch.load('delta_mid.pth')
    # delta_x = loaded_data['delta'].float() /255.0
    # mid_x = loaded_data['mid'].float() / 255.0
    # data_x = (data_x.T - mid_x) / delta_x * 2
    # data_x = data_x.T
    return data_x

def dyn_mnist(dataset, max_sv=100, WEIGHT=1, LR=1e-2, BS=64, isCls=False):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    train_err_list = []
    test_err_list = []
    sv_list = []
    # clean data
    data_sv_x, data_sv_y = dataset.get_sv_data()
    data_train_x, data_train_y = dataset.get_train_data()
    data_test_x, data_test_y = dataset.get_test_data()

    data_sv_y, data_train_y, data_test_y = preY_one_hot(data_sv_y, data_train_y, data_test_y)

    data_sv_x = preX(data_sv_x).float().to(device)
    data_sv_y = data_sv_y.float().to(device)
    data_train_x = preX(data_train_x).float()
    data_train_y = data_train_y.float()
    data_test_x = preX(data_test_x).float()
    data_test_y = data_test_y.float()

    weight_dim = min(data_train_x.shape[0], max(20, data_train_x.shape[0] // 1))
    weight_last = (torch.tensor(WEIGHT).float() / float(data_train_x.shape[0])
                   * torch.ones(weight_dim, data_sv_x.shape[1]))

    inner_iter = 0
    while inner_iter < 2:
        sv_num = data_sv_x.shape[1]
        # if sv_num > 100 and sv_num % 100 == 0 or sv_num > 200:
        #     inner_iter = 1
        # elif sv_num != max_sv:
        #     inner_iter = 0
        print('sv: ', sv_num)
        sv_list.append(sv_num)
        add_num = sv_num - weight_last.shape[1]
        weight_mean = torch.mean(weight_last).cpu()
        weight_ini = torch.cat((weight_last.cpu(), weight_mean*torch.ones(weight_dim, add_num)), 1)
        LABRBF_model = LAB_unroll(x_sv=data_sv_x, y_sv=data_sv_y, weight_ini=weight_ini, cla=data_train_y.shape[0])
        LABRBF_model.float().to(device)
        LABRBF_model.eval()

        start_time = time.time()
        TrainKernel(LABRBF_model, optFlag=inner_iter, LR=LR, BS=BS, isCls=isCls)
        run_time = time.time() - start_time
        print('training time:', format(run_time))
        weight_last = LABRBF_model.weight.data.detach().cpu()

        # LABRBF_model.alpha = LABRBF_model.exact_alpha()
        start_time = time.time()
        print('the train error rate loss: ')
        if isCls:
            train_err, pred_y = testFun_cls(LABRBF_model, data_train_x, data_train_y)
        else:
            train_err, pred_y = testFun_reg(LABRBF_model, data_train_x, data_train_y)
        run_time = time.time() - start_time
        print('forward time:', format(run_time))

        print('the test error rate loss: ')
        if isCls:
            test_err, _ = testFun_cls(LABRBF_model, data_test_x, data_test_y)
        else:
            test_err, _ = testFun_reg(LABRBF_model, data_test_x, data_test_y)

        train_err_list.append(train_err.cpu())
        test_err_list.append(test_err.cpu())

        tt = max_sv - sv_num
        k = min(max(20, int(sv_num / 10)), tt)
        if k <= 0:
            inner_iter = inner_iter + 1
            continue

        if not isCls:
            err_tmp = torch.norm(data_train_y - pred_y, dim=0)
            _, idx = torch.topk(err_tmp, k, dim=0)
        else:
            idx = add_new_sv_cls(data_train_y, pred_y, k)

        data_sv_x = torch.cat((data_sv_x, data_train_x[:, idx.reshape(-1)].to(device)), 1)
        data_sv_y = torch.cat((data_sv_y, data_train_y[:, idx.reshape(-1)].to(device)), 1)


    print('the final test error rate loss: ')
    if isCls:
        test_err, pred_y = testFun_cls(LABRBF_model, data_test_x, data_test_y)
    else:
        test_err, pred_y = testFun_reg(LABRBF_model, data_test_x, data_test_y)


    current_time = datetime.now().strftime("%H_%M_%S")
    plt.savefig(f"loss_{current_time}.png")
    fig, ax1 = plt.subplots()
    ax1.plot(test_err_list, color='k', marker='o', label='Test',
             linestyle='--', linewidth=3, markeredgewidth=6)
    ax1.plot(train_err_list, color='k', marker='*', label='Train',
             linestyle='-', linewidth=3, markeredgewidth=6)
    plt.xlabel('Iteration', fontsize=14)
    ax1.set_ylabel('Accuracy', fontsize=14)
    ax2 = ax1.twinx()
    ax2.plot(sv_list, color='k', marker='1', label='# S.V.',
             linestyle=':', linewidth=3, markeredgewidth=6)
    ax2.set_ylabel('# S.V.', fontsize=14)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=12)
    # plt.show()
    plt.savefig(f"ERR_{current_time}.png")
    return test_err


def add_new_sv_cls(train_y, pred_y, sv_num, cls=10):
    pred_y_sign = torch.argmax(pred_y, dim=0)
    train_y_sign = train_y.argmax(dim=0)
    different_indices = (pred_y_sign != train_y_sign).nonzero(as_tuple=False).squeeze()
    if len(different_indices) == 0:
        return []
    idx = np.empty((0))
    for label in range(cls):
        class_indices = np.where(train_y_sign[different_indices] == label)[0]
        class_indices = np.concatenate([different_indices[class_indices],  np.where(train_y_sign == label)[0]])

        idx_tmp = np.random.choice(class_indices, max(1, sv_num // cls), replace=False)
        idx = np.concatenate([idx, idx_tmp])
    if len(idx) > sv_num:
        idx = idx[:sv_num]
    return idx

