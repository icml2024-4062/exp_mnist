import os
import torch
import scipy.io as sio
from torch.utils.data import Dataset
import scipy.io as scio
import numpy as np
from sklearn.datasets import load_breast_cancer, fetch_kddcup99
import gzip
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, PCAM

def select_uniformly(sorted_list, N):
    length = len(sorted_list)
    interval = length // N
    selected_items = [sorted_list[i * interval] for i in range(N)]

    return selected_items

def y_label_based_sv(y, required_sv):
    _, index = torch.sort(y)
    sv_id = []
    while len(sv_id) < required_sv:
        rest_num = required_sv - len(sv_id)
        step = len(index) // rest_num
        sv_id = index[::step]
        indices_to_remove = torch.isin(index, sv_id)
        index = index[~indices_to_remove]
    return sv_id

class Cifar5M(Dataset):

    def __init__(self, name, rate=10, sv_per_cls=10):
        self.dataFolder = '/users/sista/fhe/no_backup/bigDataset/'

        dir = self.dataFolder + name
        data = np.load(dir)
        self.data_x = torch.from_numpy(data['X'].squeeze().T)
        self.data_y = torch.from_numpy(data['Y'])
        self.test_num = 3000
        self.train_num = rate * 1000
        self.val_num = rate * 10
        idx = np.random.permutation(range(len(self.data_y)))
        self.test_id = idx[0:self.test_num]
        self.train_id = idx[self.test_num:self.test_num + self.train_num]
        self.val_id = idx[self.test_num + self.train_num:self.test_num+self.train_num+self.val_num]

        class_indices = {label: np.where(self.data_y[self.train_id] == label)[0] for label in range(10)}
        idx = np.concatenate(
            [np.random.choice(class_indices[label], sv_per_cls, replace=False) for label in range(10)])
        self.sv_id = self.train_id[idx]
        self.sv_num = len(self.sv_id)

        print('sv_num: ', self.sv_num, 'val_num: ', self.val_num, 'train_num: ', self.train_num,
              'test_num: ', self.test_num)

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return self.train_num

    def get_sv_data(self):
        return self.data_x[:, self.sv_id], self.data_y[self.sv_id]

    def get_train_data(self):
        return self.data_x[:, self.train_id], self.data_y[self.train_id]

    def get_val_data(self):
        return self.data_x[:, self.val_id], self.data_y[self.val_id]

    def get_test_data(self):
        return self.data_x[:, self.test_id], self.data_y[self.test_id]

class CifarStar(Dataset):

    def __init__(self, required_sv=100):
        self.dataFolder = '/users/sista/fhe/no_backup/bigDataset/'

        dir = self.dataFolder + 'cifar_resnet34_fea_part0.npz'
        data = np.load(dir)
        self.train_x = torch.from_numpy(data['X'].squeeze().T)
        self.train_y = torch.from_numpy(data['Y'])
        self.train_num = len(self.train_y)

        dir = self.dataFolder + 'cifar_resnet34_test_part0.npz'
        data = np.load(dir)
        self.test_x = torch.from_numpy(data['X'].squeeze().T)
        self.test_y = torch.from_numpy(data['Y'])
        self.test_num = len(self.test_y)

        idx = y_label_based_sv(self.train_y, required_sv)
        self.sv_id = np.arange(self.train_num)[idx]
        self.sv_num = required_sv

        print('sv_num: ', self.sv_num, 'train_num: ', self.train_num,
              'test_num: ', self.test_num)

    def __getitem__(self, index):
        return self.train_x[index], self.train_y[index]

    def __len__(self):
        return self.train_num

    def get_sv_data(self):
        return self.train_x[:, self.sv_id], self.train_y[self.sv_id]

    def get_train_data(self):
        return self.train_x, self.train_y

    def get_test_data(self):
        return self.test_x, self.test_y

class CifarMy(Dataset):

    def __init__(self, required_sv=100):
        # self.dataFile = 'D:/code/learningkernel_svm/cifar'
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert PIL Image to PyTorch Tensor
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),  # Normalize to [-1, 1]
        ])

        # Load the CIFAR-10 training dataset
        cifar_train_dataset = CIFAR10(root='/users/sista/fhe/no_backup/dataset/cifar', train=True, download=False, transform=transform)

        # Create a DataLoader for the training dataset
        batch_size = 64
        cifar_train_dataloader = DataLoader(cifar_train_dataset, batch_size=batch_size, shuffle=True)
        xx = []
        yy = []
        for batch_x, batch_y in cifar_train_dataloader:
            xx.append(batch_x.reshape(len(batch_y), -1))
            yy.append(batch_y)
        self.train_x = torch.cat(xx, dim=0).T
        self.train_y = torch.cat(yy, dim=0)

        # Load the CIFAR-10 test dataset
        cifar_test_dataset = CIFAR10(root='/users/sista/fhe/no_backup/dataset/cifar', train=False, download=False, transform=transform)
        cifar_test_dataloader = DataLoader(cifar_test_dataset, batch_size=batch_size, shuffle=False)
        xx = []
        yy = []
        for batch_x, batch_y in cifar_test_dataloader:
            xx.append(batch_x.reshape(len(batch_y), -1))
            yy.append(batch_y)
        self.test_x = torch.cat(xx, dim=0).T
        self.test_y = torch.cat(yy, dim=0)

        self.train_num = self.train_x.shape[1]
        self.test_num = self.test_x.shape[1]

        idx = y_label_based_sv(self.train_y, required_sv)
        self.sv_id = np.arange(self.train_num)[idx]
        self.sv_num = required_sv

        print('sv_num: ', self.sv_num, 'train_num: ', self.train_num, 'test_num: ',
              self.test_num)

    def __getitem__(self, index):
        return self.train_x[index], self.train_y[index]

    def __len__(self):
        return self.train_num

    def get_sv_data(self):
        return self.train_x[:, self.sv_id], self.train_y[self.sv_id]

    def get_train_data(self):
        return self.train_x, self.train_y

    def get_test_data(self):
        return self.test_x, self.test_y


class BreastCancer(Dataset):

    def __init__(self, required_sv=10):
        data = load_breast_cancer()
        X, y = data.data, data.target
        self.data_x = torch.transpose(torch.from_numpy(X), 0, 1).float()
        print(self.data_x.shape)
        # for fea_id in range(self.data_x.shape[0]):
        #     self.data_x[fea_id, :] = self.data_x[fea_id, :] / torch.max(torch.abs(self.data_x[fea_id, :]))

        self.data_y = torch.from_numpy(y).long()

        self.num_samples = self.data_x.shape[1]
        self.train_num = int(np.ceil(0.8 * self.num_samples))
        self.test_num = self.num_samples - self.train_num
        ind_tmp = np.random.permutation(self.num_samples)
        self.train_id = ind_tmp[0: self.train_num]
        self.test_id = ind_tmp[self.train_num:]

        idx = y_label_based_sv(self.data_y[self.train_id], required_sv)
        self.sv_id = self.train_id[idx]
        self.sv_num = required_sv

        print('sv_num: ', self.sv_num, 'train_num: ', self.train_num, 'test_num: ',
              self.test_num)

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return self.num_samples

    def get_sv_data(self):
        return self.data_x[:, self.sv_id], self.data_y[self.sv_id]

    def get_train_data(self):
        return self.data_x[:, self.train_id], self.data_y[self.train_id]

    def get_test_data(self):
        return self.data_x[:, self.test_id], self.data_y[self.test_id]


class myPCAM(Dataset):

    def __init__(self, required_sv=10):
        transform = transforms.Compose([
            transforms.ToTensor()#,
            #transforms.Normalize(mean=torch.tensor([0.2860]), std=torch.tensor([0.3530]))
        ])

        batch_size = 64
        mnist_train_dataset = PCAM(root='/users/sista/fhe/no_backup/dataset', split='train', download=True, transform=transform)
        mnist_train_dataloader = DataLoader(mnist_train_dataset, batch_size=batch_size, shuffle=True)
        xx = []
        yy = []
        for batch_x, batch_y in mnist_train_dataloader:
            xx.append(batch_x.reshape(len(batch_y), -1))
            yy.append(batch_y)
        self.train_x = torch.cat(xx, dim=0).T
        self.train_y = torch.cat(yy, dim=0)

        mnist_test_dataset = PCAM(root='/users/sista/fhe/no_backup/dataset', split='test', download=True, transform=transform)
        mnist_test_dataloader = DataLoader(mnist_test_dataset, batch_size=batch_size, shuffle=False)
        xx = []
        yy = []
        for batch_x, batch_y in mnist_test_dataloader:
            xx.append(batch_x.reshape(len(batch_y), -1))
            yy.append(batch_y)
        self.test_x = torch.cat(xx, dim=0).T
        self.test_y = torch.cat(yy, dim=0)

        self.train_num = self.train_x.shape[1]
        self.test_num = self.test_x.shape[1]

        idx = y_label_based_sv(self.train_y, required_sv)
        self.sv_id = np.arange(self.train_num)[idx]
        self.sv_num = required_sv

        print('sv_num: ', self.sv_num, 'train_num: ', self.train_num, 'test_num: ',
              self.test_num)

    def __getitem__(self, index):
        return self.train_x[index], self.train_y[index]

    def __len__(self):
        return self.train_num

    def get_sv_data(self):
        return self.train_x[:, self.sv_id], self.train_y[self.sv_id]

    def get_train_data(self):
        return self.train_x, self.train_y

    def get_test_data(self):
        return self.test_x, self.test_y


class myFashionMnist(Dataset):

    def __init__(self, required_sv=10):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor([0.2860]), std=torch.tensor([0.3530]))
        ])

        batch_size = 64
        mnist_train_dataset = FashionMNIST(root='/users/sista/fhe/no_backup/dataset', train=True, download=True, transform=transform)
        mnist_train_dataloader = DataLoader(mnist_train_dataset, batch_size=batch_size, shuffle=True)
        xx = []
        yy = []
        for batch_x, batch_y in mnist_train_dataloader:
            xx.append(batch_x.reshape(len(batch_y), -1))
            yy.append(batch_y)
        self.train_x = torch.cat(xx, dim=0).T
        self.train_y = torch.cat(yy, dim=0)

        mnist_test_dataset = FashionMNIST(root='/users/sista/fhe/no_backup/dataset', train=False, download=False, transform=transform)
        mnist_test_dataloader = DataLoader(mnist_test_dataset, batch_size=batch_size, shuffle=False)
        xx = []
        yy = []
        for batch_x, batch_y in mnist_test_dataloader:
            xx.append(batch_x.reshape(len(batch_y), -1))
            yy.append(batch_y)
        self.test_x = torch.cat(xx, dim=0).T
        self.test_y = torch.cat(yy, dim=0)

        self.train_num = self.train_x.shape[1]
        self.test_num = self.test_x.shape[1]

        idx = y_label_based_sv(self.train_y, required_sv)
        self.sv_id = np.arange(self.train_num)[idx]
        self.sv_num = required_sv

        print('sv_num: ', self.sv_num, 'train_num: ', self.train_num, 'test_num: ',
              self.test_num)

    def __getitem__(self, index):
        return self.train_x[index], self.train_y[index]

    def __len__(self):
        return self.train_num

    def get_sv_data(self):
        return self.train_x[:, self.sv_id], self.train_y[self.sv_id]

    def get_train_data(self):
        return self.train_x, self.train_y

    def get_test_data(self):
        return self.test_x, self.test_y

class myMnist(Dataset):

    def __init__(self, required_sv=10):
        # self.dataFile = 'D:/code/learningkernel_svm/cifar'
        transform = transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.RandomRotation(degrees=10),
            # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            # transforms.RandomResizedCrop(size=(28, 28), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor([0.1307]),
                 std=torch.tensor([0.3081]))
        ])

        batch_size = 64
        mnist_train_dataset = MNIST(root='/users/sista/fhe/no_backup/dataset', train=True, download=False, transform=transform)
        mnist_train_dataloader = DataLoader(mnist_train_dataset, batch_size=batch_size, shuffle=True)
        xx = []
        yy = []
        for batch_x, batch_y in mnist_train_dataloader:
            xx.append(batch_x.reshape(len(batch_y), -1))
            yy.append(batch_y)
        self.train_x = torch.cat(xx, dim=0).T
        self.train_y = torch.cat(yy, dim=0)

        mnist_test_dataset = MNIST(root='/users/sista/fhe/no_backup/dataset', train=False, download=False, transform=transform)
        mnist_test_dataloader = DataLoader(mnist_test_dataset, batch_size=batch_size, shuffle=False)
        xx = []
        yy = []
        for batch_x, batch_y in mnist_test_dataloader:
            xx.append(batch_x.reshape(len(batch_y), -1))
            yy.append(batch_y)
        self.test_x = torch.cat(xx, dim=0).T
        self.test_y = torch.cat(yy, dim=0)

        self.train_num = self.train_x.shape[1]
        self.test_num = self.test_x.shape[1]

        idx = y_label_based_sv(self.train_y, required_sv)
        self.sv_id = np.arange(self.train_num)[idx]
        self.sv_num = required_sv

        print('sv_num: ', self.sv_num, 'train_num: ', self.train_num, 'test_num: ',
              self.test_num)

    def __getitem__(self, index):
        return self.train_x[index], self.train_y[index]

    def __len__(self):
        return self.train_num

    def get_sv_data(self):
        return self.train_x[:, self.sv_id], self.train_y[self.sv_id]

    def get_train_data(self):
        return self.train_x, self.train_y

    def get_test_data(self):
        return self.test_x, self.test_y

class MnistAll(Dataset):

    def __init__(self, class_list=None, required_sv=100):
        if class_list is None:
            class_list = [0, 1]
        self.dataFile = '/users/sista/fhe/no_backup/dataset/MNIST'
        data_tmp = scio.loadmat(self.dataFile + '/mnist_all.mat')
        self.data_x_train = torch.empty((data_tmp['train0'].shape[1],0),dtype=torch.float)
        self.data_x_test = self.data_x_train
        self.data_y_train = torch.empty(0, dtype=torch.int64)
        self.data_y_test = self.data_y_train
        # self.data_x_train = torch.transpose(torch.from_numpy(data_tmp['train0']), 0, 1).float()
        # self.data_x_test = torch.transpose(torch.from_numpy(data_tmp['test0']), 0, 1).float()
        # self.data_y_train = 0 * torch.ones(self.data_x_train.shape[1], dtype=torch.int)
        # self.data_y_test = 0 * torch.ones(self.data_x_test.shape[1], dtype=torch.int)
        for ii in class_list:
            name = 'train'+str(ii)
            tmp_x = torch.transpose(torch.from_numpy(data_tmp[name]), 0, 1).float()
            self.data_x_train = torch.cat((self.data_x_train, tmp_x), 1)
            tmp_y = ii * torch.ones(tmp_x.shape[1], dtype=torch.int64)
            self.data_y_train = torch.cat((self.data_y_train, tmp_y), 0)
            name = 'test'+str(ii)
            tmp_x = torch.transpose(torch.from_numpy(data_tmp[name]), 0, 1).float()
            self.data_x_test = torch.cat((self.data_x_test, tmp_x), 1)
            tmp_y = ii * torch.ones(tmp_x.shape[1], dtype=torch.int64)
            self.data_y_test = torch.cat((self.data_y_test, tmp_y), 0)
        # self.data_y_test = self.data_y_test.view(1,-1)
        # self.data_y_train = self.data_y_train.view(1, -1)
        self.train_num = self.data_x_train.shape[1]
        self.test_num = self.data_x_test.shape[1]

        # self.train_id = range(0, self.train_num)
        self.sv_id = select_uniformly(range(0, self.train_num), required_sv)

        #
        # self.val_id = list(set(self.train_id).difference(set(self.sv_id)))[::10]
        # self.train_id = list(set(self.train_id).difference(set(self.val_id)))

        print('sv_num: ', len(self.sv_id), 'train_num: ',
              self.train_num, 'test_num: ', self.test_num)

    def __getitem__(self, index):
        return self.data_x_train[index], self.data_y_train[index]

    def __len__(self):
        return self.train_num

    def get_sv_data(self):
        return self.data_x_train[:, self.sv_id], self.data_y_train[self.sv_id]

    def get_train_data(self):
        return self.data_x_train, self.data_y_train

    def get_test_data(self):
        return self.data_x_test, self.data_y_test

class WineQuality(Dataset):

    def __init__(self,class_list):
        self.dataFile = 'D:/code/learningkernel_svm/winequality'
        data_tmp = scio.loadmat(self.dataFile + '/winequality.mat')
        tmp_y = torch.from_numpy(data_tmp['winequalitywhiteY'])
        tmp_x = torch.from_numpy(data_tmp['winequalitywhiteX']).T.to(torch.float)
        self.data_x = torch.empty((tmp_x.shape[0], 0))
        self.data_y = torch.empty((0,1),dtype=torch.int32)
        self.train_id = np.empty(0, dtype=np.int32)
        self.test_id = np.empty(0, dtype=np.int32)
        cnt = 0
        for cls in class_list:
            idx = torch.nonzero(tmp_y.reshape(-1) == cls).squeeze()
            self.data_x = torch.cat((self.data_x, tmp_x[:,idx]), 1)
            self.data_y = torch.cat((self.data_y, tmp_y[idx,:]), 0)
            num_train = int(np.ceil(0.8*len(idx)))
            idx_tmp = np.random.permutation(len(idx)) + cnt
            self.train_id = np.concatenate((self.train_id, idx_tmp[0:num_train]), 0)
            self.test_id = np.concatenate((self.test_id, idx_tmp[num_train:]), 0)
            cnt = cnt + len(idx)
        self.data_y = self.data_y - torch.min(self.data_y)
        self.num_samples = self.data_x.shape[1]
        self.train_num = len(self.train_id)
        self.test_num = self.num_samples - self.train_num
        self.sv_id = self.train_id[::50]

        print('sv_num: ', len(self.sv_id), 'train_num: ',
              len(self.train_id), 'test_num: ', self.test_num)

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return self.num_samples

    def get_sv_data(self):
        return self.data_x[:, self.sv_id], self.data_y[self.sv_id,:]

    def get_train_data(self):
        return self.data_x[:,self.train_id], self.data_y[self.train_id,:]

    def get_test_data(self):
        return self.data_x[:,self.test_id], self.data_y[self.test_id,:]


def load_mnist(path, kind='train', vec=True):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)
    if vec:
        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)
    else:
        with gzip.open(images_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 28, 28, 1)

    return images, labels

class FashionMnist(Dataset):

    def __init__(self, required_sv=20):
        self.dataFile = '/users/sista/fhe/no_backup/dataset/fashionMnist'
        X_train, y_train = load_mnist(self.dataFile, kind='train')
        X_test, y_test = load_mnist(self.dataFile, kind='t10k')

        self.train_x = torch.from_numpy(np.copy(X_train.T))
        self.train_y = torch.from_numpy(np.copy(y_train)).long()
        self.test_x = torch.from_numpy(np.copy(X_test.T))
        self.test_y = torch.from_numpy(np.copy(y_test)).long()

        self.train_num = self.train_x.shape[1]
        self.test_num = self.test_x.shape[1]

        idx = y_label_based_sv(self.train_y, required_sv)
        self.sv_id = np.arange(self.train_num)[idx]
        self.sv_num = required_sv

        print('sv_num: ', self.sv_num, 'train_num', self.train_num, 'test_num: ',
              self.test_num)

    def __getitem__(self, index):
        return self.train_x[index], self.train_y[index]

    def __len__(self):
        return self.num_samples

    def get_sv_data(self):
        return self.train_x[:, self.sv_id], self.train_y[self.sv_id]

    def get_train_data(self):
        return self.train_x, self.train_y

    def get_test_data(self):
        return self.test_x, self.test_y

