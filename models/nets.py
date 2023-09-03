import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.misc import euclidean_metric, count_acc, compute_proto_tc
from torch.nn.utils.weight_norm import WeightNorm
from torch.distributions import Normal, kl_divergence
from scipy.spatial.distance import pdist, cdist
from models.res12 import Res12
from models.wrn28 import Wrn28
import numpy as np

def intra_ter_dist(X, Y, way):
    intra = 0.0
    centers = []
    for yk in range(way):
        idk = np.where(Y == yk)[0]
        meanc = X[idk].mean(0)
        intra = intra + pdist(X[idk]).mean()
        # intra = intra + cdist(X[idk], meanc.reshape(1,-1)).mean()
        centers.append(meanc)
    intra = intra / way
    centers = np.stack(centers, axis=0)
    inter = pdist(centers).mean()

    return intra, inter

def intra_dist_class(X):
    # X and Y should be numpy
    # meanc = X.mean(0)
    # intra = cdist(X, meanc.reshape(1,-1)).mean()
    intra = pdist(X).mean()
    return intra

def classes_items(logit1, logit2, logit3, label_q, X_1, X_2, X_3, way):
    Acclist1 = []
    Acclist2 = []
    Acclist3 = []
    Intralist1 = []
    Intralist2 = []
    Intralist3 = []

    for ki in range(way):
        idd = torch.where(label_q == ki)[0]
        Acclist1.append(count_acc(logit1[idd], label_q[idd]) * 100)
        Acclist2.append(count_acc(logit2[idd], label_q[idd]) * 100)
        Acclist3.append(count_acc(logit3[idd], label_q[idd]) * 100)
        idd = idd.cuda().data.cpu().numpy()
        intra_1 = intra_dist_class(X_1[idd])
        Intralist1.append(intra_1)
        intra_2 = intra_dist_class(X_2[idd])
        Intralist2.append(intra_2)
        intra_3 = intra_dist_class(X_3[idd])
        Intralist3.append(intra_3)

    return Acclist1, Acclist2, Acclist3, Intralist1, Intralist2, Intralist3


def compute_class_dstr(mu, logvar, lb, args):
    var = torch.exp(logvar)
    M = []
    C = []
    if args.merge_dstr == 'mean':
        for y in torch.unique(lb):
            id = torch.where(lb == y)[0]
            mu_ = mu[id].mean(0)
            var_ = var[id].mean(0)
            M.append(mu_)
            C.append(var_)
    elif args.merge_dstr == 'merge':
        for y in torch.unique(lb):
            id = torch.where(lb == y)[0]
            mu_ = mu[id]
            var_ = var[id]
            Cmean = ((var_ ** (-1)).sum(0)) ** (-1) #Csum = ((var_ ** (-1)).mean(0)) ** (-1)
            Mnew = Cmean * (((var_ ** (-1)) * mu_).sum(0))
            Csum = ((var_ ** (-1)).mean(0)) ** (-1)
            M.append(Mnew)
            C.append(Csum)
    M = torch.stack(M, 0)
    C = torch.stack(C, 0)
    return M, C


class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias=False)
        WeightNorm.apply(self.L, 'weight', dim=0)
        if outdim <= 200:
            self.scale_factor = 2
        else:
            self.scale_factor = 10

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        cos_dist = self.L(x_normalized)
        scores = self.scale_factor * (cos_dist)
        return scores


class Model(nn.Module):
    """The class for outer loop."""
    def __init__(self, args, mode='pre', num_cls=64):
        super().__init__()
        self.args = args
        self.mode = mode
        if self.args.dataset == 'cub':
            sdim = 312
        else:
            sdim = 300

        if self.args.WEmodel == 'clip':
            sdim = 512

        if self.args.model_type == 'res12':
            self.encoder = Res12()
            z_dim = 640
        elif self.args.model_type == 'wrn28':
            self.encoder = Wrn28()
            z_dim = 640
        self.z_dim = z_dim
        self.sdim = sdim
        self.latentdim = self.args.latentdim

        if self.mode == 'pre':
            self.pre_fc = distLinear(z_dim, num_cls)
            self.rot_fc = nn.Linear(z_dim, 4)
        elif self.mode == 'mvc':
            self.contloss = torch.nn.CrossEntropyLoss()

            self.mun = nn.Linear(self.z_dim, self.sdim)
            self.varn = nn.Linear(self.z_dim, self.sdim)
            self.rect = nn.Linear(self.sdim, self.z_dim)

            self.MuNet = nn.Linear(sdim + z_dim, self.latentdim)
            self.VarNet = nn.Linear(sdim + z_dim, self.latentdim)
            self.ReNet = nn.Linear(self.latentdim, sdim + z_dim)


    def forward(self, inp):
        if self.mode == 'pre':
            return self.forward_pretrain(inp)
        elif self.mode == 'preval':
            datas, dataq = inp
            return self.forward_preval(datas, dataq)
        elif self.mode == 'mvc':
            datas, ys, sem, dataq = inp
            return self.forward_mvc(datas, ys, sem, dataq)
        else:
            raise ValueError('Please set the correct mode.')

    def forward_pretrain(self, inp):
        embedding = self.encoder(inp)
        logits = self.pre_fc(embedding)
        rot = self.rot_fc(embedding)
        return logits, rot

    def forward_preval(self, data_shot, data_query):
        proto = self.encoder(data_shot)
        proto = proto.reshape(self.args.shot, self.args.way, -1).mean(dim=0)
        query = self.encoder(data_query)
        if self.args.metric == 'ED':
            logitq = euclidean_metric(query, proto)
        elif self.args.metric == 'cos':
            x_mul = torch.matmul(query, proto.T)
            Normv = torch.mul(torch.norm(query, dim=1).unsqueeze(1), torch.norm(proto, dim=1).unsqueeze(0))
            logitq = torch.div(x_mul, Normv)
        return logitq


    def forward_mvc(self, datas, ys, sem, dataq):
        semway = sem[:self.args.way]
        datas = F.normalize(datas, 1)
        dataq = F.normalize(dataq, 1)
        sem = F.normalize(sem, 1)

        proto = compute_proto_tc(datas, ys, self.args.way)

        if self.args.metric == 'ED':
            logit0 = euclidean_metric(dataq, proto)
        elif self.args.metric == 'cos':
            x_mul = torch.matmul(dataq, proto.T)
            Normv = torch.mul(torch.norm(dataq, dim=1).unsqueeze(1), torch.norm(proto, dim=1).unsqueeze(0))
            logit0 = torch.div(x_mul, Normv)

        optimizer1 = torch.optim.Adam([
            {'params': self.mun.parameters()},
            {'params': self.varn.parameters()},
            {'params': self.rect.parameters()}
        ], lr=self.args.lr, weight_decay=1e-5)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=50, gamma=0.2)
        self.mun.train()
        self.varn.train()
        self.rect.train()
        proto1 = compute_proto_tc(datas, ys, self.args.way)
        for i in range(30):
            mu = self.mun(datas)
            mu = F.normalize(mu, 1)
            logvar = self.varn(datas)
            logvar = F.normalize(logvar, 1)
            std = torch.exp(0.5 * logvar)
            logit = mu.mm(semway.T)
            loss_con = self.contloss(logit * self.args.temprature, ys)
            dtr1 = torch.distributions.normal.Normal(mu, std)
            I = torch.ones_like(mu).type(mu.type())
            Z = torch.zeros_like(mu).type(mu.type())
            dtr2 = torch.distributions.normal.Normal(Z, I)
            loss_kl = torch.distributions.kl.kl_divergence(dtr1, dtr2).sum(-1).mean()

            latentx = dtr1.rsample()
            recX = self.rect(latentx)
            recX = F.normalize(recX, 1)

            logit1 = recX.mm(proto1.T)
            loss_rec = self.contloss(logit1 * self.args.temprature, ys)
            # p_x = torch.distributions.Normal(recX, torch.ones_like(recX))
            # loss_log = p_x.log_prob(datas).sum(-1).mean()

            loss = loss_con + loss_kl + loss_rec
            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()
            lr_scheduler.step()
        self.mun.eval()
        self.varn.eval()
        self.rect.eval()
        semq = self.mun(dataq).detach()

        if self.args.metric == 'ED':
            logit1 = euclidean_metric(semq, semway)
        elif self.args.metric == 'cos':
            x_mul = torch.matmul(semq, semway.T)
            Normv = torch.mul(torch.norm(semq, dim=1).unsqueeze(1), torch.norm(semway, dim=1).unsqueeze(0))
            logit1 = torch.div(x_mul, Normv)

        
        S_mu = self.mun(datas).detach()
        S_var = self.varn(datas).detach()

        mu1, var = compute_class_dstr(S_mu, S_var, ys, self.args)
        Sp_std = var ** 0.5
        dtr = torch.distributions.normal.Normal(semway, Sp_std)

        Sfnew = []
        Synew = []
        Ssnew = []
        for aug in range(self.args.expandnum):
            textvec = dtr.rsample()
            recX = self.rect(textvec).detach()
            recX = F.normalize(recX, 1)
            Sfnew.append(recX)
            Synew.append(ys[:self.args.way])
            Ssnew.append(semway)
        Sfnew = torch.cat(Sfnew, dim=0)
        Synew = torch.cat(Synew, dim=0)
        Ssnew = torch.cat(Ssnew, dim=0)
        
        Sdata = torch.cat([datas, Sfnew], dim=0)
        Ssem = torch.cat([sem, Ssnew], dim=0)
        Sy = torch.cat([ys, Synew], dim=0)

        inpS = torch.cat([Sdata, Ssem], dim=1)
        inpQ = torch.cat([dataq, semq], dim=1)

        optimizer2 = torch.optim.Adam([
            {'params': self.MuNet.parameters()},
            {'params': self.VarNet.parameters()},
            #{'params': self.ReNet.parameters()}
        ], lr=0.1, weight_decay=1e-5) #0.1
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=50, gamma=0.2)
        self.MuNet.train()
        self.VarNet.train()
        #self.ReNet.train()

        for i in range(30):
            mu = self.MuNet(inpS)
            logvar = self.VarNet(inpS)
            mu = F.normalize(mu, 1)
            logvar = F.normalize(logvar, 1)

            std = torch.exp(0.5 * logvar)
            dtr1 = torch.distributions.normal.Normal(mu, std)
            I = torch.ones_like(mu).type(mu.type())
            Z = torch.zeros_like(mu).type(mu.type())
            dtr2 = torch.distributions.normal.Normal(Z, I)
            loss = torch.distributions.kl.kl_divergence(dtr1, dtr2).sum(-1).mean()

            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()
            lr_scheduler.step()
        self.MuNet.eval()
        self.VarNet.eval()
        #self.ReNet.eval()

        mus = self.MuNet(inpS)
        logvars = self.VarNet(inpS)
        mus, var = compute_class_dstr(mus, logvars, ys, self.args)
        stds = var ** 0.5

        muq = self.MuNet(inpQ)
        logvarq = self.VarNet(inpQ)
        stdq = torch.exp(0.5 * logvarq)

        mus = F.normalize(mus, 1)
        stds = F.normalize(stds, 1)
        muq = F.normalize(muq, 1)
        stdq = F.normalize(stdq, 1)

        Logit = []
        for k in range(muq.size(0)):
            dbt1 = Normal(muq[k].view(1, -1), stdq[k].view(1, -1))
            dbt2 = Normal(mus, stds)
            sim = -kl_divergence(dbt1, dbt2).mean(-1)
            Logit.append(sim)
        logit2_kl = torch.stack(Logit, dim=0)

        # x_mul = torch.matmul(muq, mus.T)
        # Normv = torch.mul(torch.norm(muq, dim=1).unsqueeze(1), torch.norm(mus, dim=1).unsqueeze(0))
        # logit2_cos = torch.div(x_mul, Normv)
        # logit2_ed = euclidean_metric(muq, mus)

        return logit0, logit2_kl
