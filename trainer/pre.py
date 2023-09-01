import os.path as osp
import os
import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader.samplers import CategoriesSampler
from models.nets import Model
from utils.misc import Averager, Timer, count_acc, ensure_path, compute_confidence_interval
from tensorboardX import SummaryWriter
from dataloader.dataset_loader import DatasetLoader as Dataset
import numpy as np


class PRETrainer(object):
    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        log_base_dir = './logs/'
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)
        pre_base_dir = osp.join(log_base_dir, 'pre')
        if not osp.exists(pre_base_dir):
            os.mkdir(pre_base_dir)
        save_path1 = '_'.join([args.dataset, args.model_type])
        args.save_path = pre_base_dir + '/' + save_path1
        ensure_path(args.save_path)

        self.args = args
        # Load pretrain set
        self.trainset = Dataset('train', self.args)
        self.train_loader = DataLoader(dataset=self.trainset, batch_size=args.pre_batch_size, shuffle=True,
                                       num_workers=8, pin_memory=True)
        # Load meta-val set
        self.valset = Dataset('val', self.args)
        self.val_sampler = CategoriesSampler(self.valset.label, 500, self.args.way,
                                             self.args.shot + self.args.val_query)
        self.val_loader = DataLoader(dataset=self.valset, batch_sampler=self.val_sampler, num_workers=8,
                                     pin_memory=True)

        # Set pretrain class number
        num_class_pretrain = self.trainset.num_class

        # Build pretrain model
        self.model = Model(self.args, mode='pre', num_cls=num_class_pretrain)

        # Set optimizer
        if self.args.opt == 'SGD':
            self.optimizer = torch.optim.SGD([{'params': self.model.encoder.parameters(), 'lr': 0.0001},
                             {'params': self.model.pre_fc.parameters()}, 
                             {'params': self.model.rot_fc.parameters()}],
                            lr=self.args.pre_lr, momentum=0.9, weight_decay=self.args.pre_weight_decay,
                            nesterov=self.args.nesterov)
        elif self.args.opt == 'Adam':
            self.optimizer = torch.optim.Adam([{'params': self.model.encoder.parameters(), 'lr': 0.0001},
                             {'params': self.model.pre_fc.parameters()}, 
                             {'params': self.model.rot_fc.parameters()}],
                             lr=self.args.pre_lr, weight_decay=self.args.pre_weight_decay)
        # Set learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.scheduler_milestones,
                                               gamma=args.pre_gamma)
        
        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()

        self.model.encoder = torch.nn.DataParallel(self.model.encoder).cuda()

    def save_model(self, name):
        """The function to save checkpoints.
        Args:
          name: the name for saved checkpoint
        """
        torch.save(dict(params=self.model.encoder.state_dict()), osp.join(self.args.save_path, name + '.pth'))

    def train(self):
        # Set the pretrain log
        trlog = {}
        trlog['args'] = vars(self.args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['train_acc'] = []
        trlog['val_acc'] = []
        trlog['max_acc'] = 0.0
        trlog['max_acc_epoch'] = 0
        # Set the timer
        timer = Timer()
        # Set global count to zero
        global_count = 0
        # Set tensorboardX
        writer = SummaryWriter(comment=self.args.save_path)

        # Start pretrain
        for epoch in range(1, self.args.pre_max_epoch + 1):
            # Update learning rate
            self.lr_scheduler.step()
            # Set the model to train mode
            self.model.train()
            self.model.mode = 'pre'
            # Set averager classes to record training losses and accuracies
            train_loss_averager = Averager()
            train_acc_averager = Averager()

            # Using tqdm to read samples from train loader
            tqdm_gen = tqdm.tqdm(self.train_loader)
            for i, batch in enumerate(tqdm_gen, 1):
                self.optimizer.zero_grad()
                # Update global count number
                global_count = global_count + 1
                if torch.cuda.is_available():
                    data, label, r = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                    label = batch[1]
                    r = batch[2]
                nn, kk, cc, hh, ww = data.size()
                data = data.view(nn * kk, cc, hh, ww)
                label = label.view(-1)
                r = r.view(-1)

                if torch.cuda.is_available():
                    label = label.type(torch.cuda.LongTensor)
                else:
                    label = label.type(torch.LongTensor)
                # Output logits for model
                logits, rot = self.model(data)
                # Calculate train loss
                loss_cls = F.cross_entropy(logits, label)
                loss_rot = F.cross_entropy(rot, r)
                loss = (loss_cls + loss_rot)/2

                # Calculate train accuracy
                acc = count_acc(logits, label)
                # Write the tensorboardX records
                writer.add_scalar('data/loss', float(loss), global_count)
                writer.add_scalar('data/acc', float(acc), global_count)
                # Print loss and accuracy for this step
                tqdm_gen.set_description('Epoch {}, Loss={:.4f} Acc={:.4f}'.format(epoch, loss.item(), acc))
                # Add loss and accuracy for the averagers
                train_loss_averager.add(loss.item())
                train_acc_averager.add(acc)
                # Loss backwards and optimizer updates
                loss.backward()
                self.optimizer.step()
            # Update the averagers
            train_loss_averager = train_loss_averager.item()
            train_acc_averager = train_acc_averager.item()

            # Start validation for this epoch, set model to eval mode
            self.model.eval()
            self.model.mode = 'preval'
            # Set averager classes to record validation losses and accuracies
            val_loss_averager = Averager()
            val_acc_averager = Averager()
            # Generate the labels for test
            label = torch.arange(self.args.way).repeat(self.args.val_query)
            label_shot = torch.arange(self.args.way).repeat(self.args.shot)
            if torch.cuda.is_available():
                label_shot = label_shot.type(torch.cuda.LongTensor)
                label = label.type(torch.cuda.LongTensor)
            else:
                label_shot = label_shot.type(torch.LongTensor)
                label = label.type(torch.LongTensor)

            # Print previous information
            if epoch % 10 == 0:
                print('Best Epoch {}, Best Val acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
            # Run meta-validation
            for i, batch in enumerate(self.val_loader, 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                p = self.args.shot * self.args.way
                data_shot, data_query = data[:p], data[p:]
                logits = self.model((data_shot, data_query))
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                val_loss_averager.add(loss.item())
                val_acc_averager.add(acc)
            # Update validation averagers
            val_loss_averager = val_loss_averager.item()
            val_acc_averager = val_acc_averager.item()
            # Write the tensorboardX records
            writer.add_scalar('data/val_loss', float(val_loss_averager), epoch)
            writer.add_scalar('data/val_acc', float(val_acc_averager), epoch)
            # Print loss and accuracy for this epoch
            print('Epoch {}, Val, Loss={:.4f} Acc={:.4f}'.format(epoch, val_loss_averager, val_acc_averager))
            # Update best saved model
            if val_acc_averager > trlog['max_acc']:
                print("epoch: ", epoch)
                trlog['max_acc'] = val_acc_averager
                trlog['max_acc_epoch'] = epoch
                self.save_model('max_acc')
            # Save model every 10 epochs
            # if epoch % 10 == 0:
            #     self.save_model('epoch' + str(epoch))
            # Update the logs
            trlog['train_loss'].append(train_loss_averager)
            trlog['train_acc'].append(train_acc_averager)
            trlog['val_loss'].append(val_loss_averager)
            trlog['val_acc'].append(val_acc_averager)
            # Save log
            torch.save(trlog, osp.join(self.args.save_path, 'trlog'))

            if epoch % 10 == 0:
                print('Running Time: {}, Estimated Time: {}'.format(timer.measure(),
                                    timer.measure(epoch / self.args.pre_max_epoch)))
        writer.close()

    def eval(self):
        """The function for the meta-eval phase."""
        # Load meta-test set
        test_set = Dataset('test', self.args)
        sampler = CategoriesSampler(test_set.label, 600, self.args.way, self.args.shot + self.args.val_query)
        loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
        # Set test accuracy recorder
        test_acc_record = np.zeros((600,))

        # Load model for test phase
        model_dict = self.model.state_dict()
        state = torch.load(osp.join(self.args.save_path, 'max_acc' + '.pth'))['params']
        state = {'encoder.' + k: v for k, v in state.items()}
        state = {k: v for k, v in state.items() if k in model_dict}
        # print('model_dict: ', model_dict.keys())
        # print('------------------------------')
        # print('state: ', state.keys())
        model_dict.update(state)
        self.model.load_state_dict(model_dict)

        # Set model to eval mode
        self.model.eval()
        self.model.mode = 'preval'
        # Set accuracy averager
        ave_acc = Averager()

        # Generate labels
        label = torch.arange(self.args.way).repeat(self.args.val_query)
        label_shot = torch.arange(self.args.way).repeat(self.args.shot)
        if torch.cuda.is_available():
            label = label.type(torch.cuda.LongTensor)
            label_shot = label_shot.type(torch.cuda.LongTensor)
        else:
            label_shot = label_shot.type(torch.LongTensor)
            label = label.type(torch.LongTensor)
        # Start test
        for i, batch in enumerate(loader, 1):
            if torch.cuda.is_available():
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]
            p = self.args.shot * self.args.way
            data_shot, data_query = data[:p], data[p:]
            logit = self.model((data_shot, data_query))

            acc = count_acc(logit, label)
            ave_acc.add(acc)

            test_acc_record[i - 1] = acc
            if i % 100 == 0:
                print('batch {}: {:.2f} '.format(i, ave_acc.item() * 100))

        # Calculate the confidence interval, update the logs
        m, pm = compute_confidence_interval(test_acc_record)
        print('Test Acc {:.4f} + {:.4f}'.format(m, pm))