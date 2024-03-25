import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.checkpoint as checkpoint


class LAB_unroll(nn.Module):
    def __init__(self, x_sv, y_sv, weight_ini, cla=1, lamda=0.0001):
        super(LAB_unroll, self).__init__()

        self.num_sv = x_sv.shape[1]
        self.feature_dim = x_sv.shape[0]
        self.device = x_sv.device
        self.lamda = torch.tensor(lamda).float()
        self.cla = cla
        self.alpha = y_sv.reshape(self.cla, self.num_sv) #torch.ones(self.cla, self.num_sv)
        self.x_sv = x_sv
        self.y_sv = y_sv.reshape(self.cla, self.num_sv)
        self.center_id = range(self.num_sv)[::10]
        self.beta = torch.mean(self.y_sv, dim=1, keepdim=True)
        self.weight = torch.nn.Parameter(torch.FloatTensor(weight_ini.shape), requires_grad=True)
        # self.weight = torch.nn.Parameter(torch.FloatTensor(self.feature_dim, self.num_sv), requires_grad=True)
        self.weight.data = weight_ini.to(self.device)
        self.weight_dim = weight_ini.shape[0]
        self.scale = self.feature_dim // self.weight_dim
        self.rest = self.feature_dim % self.scale
        self.scale_mat = torch.eye(self.feature_dim, device=self.device, dtype=torch.float32)
        # self.scale_mat = torch.zeros((self.weight_dim, self.feature_dim), device=self.device, dtype=torch.float32)
        # for ii in range(self.weight_dim):
        #     for jj in range(self.feature_dim):
        #         if jj >= self.scale * ii and jj < self.scale * (ii+1):
        #             self.scale_mat[ii, jj] = 1.0
        # self.scale_mat[-1, -self.rest:] = 1.0

        # self.alpha = self.exact_alpha()

        print('Number of trainable parameters:', sum(p.numel() for p in self.parameters() if p.requires_grad) )


    def forward(self, x_train):
        assert x_train.shape[0] == self.x_sv.shape[0], 'but found {} and {}'.format(x_train.shape[0],
                                                                                     self.x_sv.shape[0])

        self.alpha = self.update_alpha()

        x_train_kernel = Gau_kernel_theta2(x_train, self.x_sv, self.weight, self.scale_mat)
        # x_train_kernel = Lap_kernel_theta(x_train, self.x_sv, self.weight, self.scale_mat)

        y_pred_train = self.alpha @ x_train_kernel + self.beta  # (cla, N)

        return y_pred_train

    def update_alpha(self):
        al_tmp = self.alpha.detach()

        if self.training:
            bs = 10
            if self.num_sv > bs:
                cnt = 0
            #     ker_list = []
            #     while cnt < self.num_sv:
            #         if cnt + bs < self.num_sv:
            #             ker_list.append(Gau_kernel_theta2(self.x_sv, self.x_sv[:, cnt:cnt+bs],
            #                                               self.weight[:, cnt:cnt+bs], self.scale_mat))
            #         else:
            #             ker_list.append(Gau_kernel_theta2(self.x_sv, self.x_sv[:, cnt:],
            #                                               self.weight[:, cnt:], self.scale_mat))
            #         cnt = cnt + bs
            #     x_sv_kernel = torch.cat(ker_list)
            # else:
            #     x_sv_kernel = Gau_kernel_theta2(self.x_sv, self.x_sv, self.weight, self.scale_mat)
            x_sv_kernel = Lap_kernel_theta(self.x_sv, self.x_sv, self.weight, self.scale_mat)

            ele1 = self.y_sv - self.beta  # y_mean
            ele2 = 1 * x_sv_kernel + self.lamda * torch.eye(self.num_sv, self.num_sv,
                                                            device=self.device, dtype=torch.float)

            al_tmp = torch.matmul(ele1, torch.linalg.inv(ele2)) * 1
            # # HH = nystrom_approximation(ele2@ele2.T, 500)
            # # HH = torch.linalg.inv(ele2@ele2.T).detach()
            # tt = []
            #--------------------
            # max_iter = 20
            # lr = 1.5 / ele2.detach().norm()**2
            # velocity = torch.zeros(al_tmp.shape, device=self.device, dtype=torch.float)
            # for ii in range(max_iter):
            #     # t1 = torch.norm(al_tmp @ ele2 - ele1).detach().cpu()
            #     # tt.append(t1)
            #     gg = self.grad(ele2, ele1, al_tmp)
            #     # al_tmp = al_tmp - 1 * (gg @ HH)
            #     velocity = velocity * 0.5 - lr * gg
            #     al_tmp = al_tmp + velocity

        return al_tmp

    def exact_alpha(self):
        # qq = 1000
        x_sv_kernel = Lap_kernel_theta(self.x_sv, self.x_sv, self.weight).detach()
        ele2 = 1 * x_sv_kernel + self.lamda * torch.eye(self.num_sv, self.num_sv,
                                                        device=self.device, dtype=torch.float)
        ele1 = self.y_sv.reshape(1, self.num_sv) - self.beta.repeat(1, self.num_sv)  # y_mean
        HH_inv = torch.linalg.inv(ele2 @ ele2.T)
        alpha = ele1 @ HH_inv
        # uu, ss, vv = torch.linalg.svd(ele2 @ ele2.T)
        # HH_sqrt = torch.diag(1./torch.sqrt(ss[0:qq])) @ vv[0:qq, :]
        return alpha

    def grad(self, AA, BB, x):
        gg = (x @ AA - BB) @ AA.T
        return gg

    @staticmethod
    def mae_loss(pred, target):
        # MAE
        loss = (torch.abs(pred.reshape(-1) - target.reshape(-1))).sum() / target.shape[0]
        return loss

    @staticmethod
    def rsse_loss(pred, target):
        # RSSE
        tmp = ((target.reshape(-1) - target.mean()) ** 2).sum()
        loss = ((pred.reshape(-1) - target.reshape(-1)) ** 2).sum() / tmp
        return loss

    @staticmethod
    def mse_loss(pred, target):
        # MSE
        loss = ((pred.reshape(-1) - target.reshape(-1)) ** 2).sum() / target.shape[0]
        return loss

    @staticmethod
    def rmse_loss(pred, target):
        # RMSE
        loss = ((pred.reshape(-1) - target.reshape(-1)) ** 2).sum() / target.shape[0]
        loss = torch.sqrt(loss)
        return loss

    @staticmethod
    def cls_loss(pred, target):
        pred_cls = pred.argmax(dim=0)
        acc = (target == pred_cls).sum() / target.shape[0]
        return acc

def euclidean_dis_theta(x, x_sv, w):
    x_sv = (x_sv * w).T  # N_sv, F
    x_sv_norm = torch.sum(x_sv ** 2, dim=1, keepdim=True)  # N_sv, 1
    inner_prod = (x_sv * w.T) @ x  # N_sv, N
    x = x.unsqueeze(0)  # 1,F,N
    w = w.T.unsqueeze(2)  # N_sv,F,1
    x_norm = torch.sum((x * w) ** 2, dim=1, keepdim=False)  # N_sv, N
    dis = x_sv_norm - 2 * inner_prod + x_norm
    return dis

def Gau_kernel_theta(x, x_sv, w, scale_mat):
    f_dim = x_sv.shape[0]
    w = (w.T @ scale_mat).T
    kernel_mat = torch.exp(-1 * euclidean_dis_theta(x, x_sv, w))
    return kernel_mat

def Gau_kernel_theta2(x, x_sv, w, scale_mat):
    f_dim = np.sqrt(x_sv.shape[0])
    x_sv = x_sv.T.unsqueeze(2)
    x = x.unsqueeze(0)
    w = w.T.unsqueeze(2)
    dis_mat = torch.sum(w * (scale_mat @ (x - x_sv)**2) / f_dim, dim=1, keepdim=False)
    kernel_mat = torch.exp(-1 * dis_mat)
    return kernel_mat


def Lap_kernel_theta(x, x_sv, w, scale_mat):
    f_dim = np.sqrt(x_sv.shape[0])
    x_sv = x_sv.T.unsqueeze(2)
    x = x.unsqueeze(0)
    w = (w.T @ scale_mat).unsqueeze(2)
    dis_mat = torch.norm(w * (x - x_sv), dim=1, keepdim=False)
    # w = w.T.unsqueeze(2)
    # dis_mat = torch.norm(w * (scale_mat @ (x - x_sv).abs()) / 1, dim=1, keepdim=False)
    kernel_mat = torch.exp(-1 * dis_mat)
    return kernel_mat

def nystrom_approximation(A, k):
    # A is the original matrix (torch.Tensor)
    m, _ = A.shape
    indices = torch.randperm(m)[:k]

    C = A[indices, :]
    W = A[:, indices][indices, :]

    # Approximate inverse
    W_inv = torch.inverse(W)
    A_approx = C.T @ W_inv @ C

    return A_approx
