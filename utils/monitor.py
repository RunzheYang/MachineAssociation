import visdom
import torch


class Plotter(object):

    def __init__(self, name=''):
        self.vis = visdom.Visdom()
        self.name = name
        self.loss_window = None
        self.acc_window = None

    def update_loss(self, cnt, loss):
        if self.loss_window == None:
            self.loss_window = self.vis.line(
                X=torch.Tensor([cnt]).cpu(),
                Y=torch.Tensor([loss]).cpu(),
                opts=dict(xlabel='iteration',
                          ylabel='nll loss',
                          title='Training Loss - ' + self.name,
                          legend=['Loss']))
        else:
            self.vis.line(
                X=torch.Tensor([cnt]).cpu(),
                Y=torch.Tensor([loss]).cpu(),
                win=self.loss_window,
                update='append')

    def update_acc(self, cnt, train_acc, val_acc):
        if self.acc_window == None:
            self.acc_window = self.vis.line(X=torch.Tensor([cnt]).cpu(),
                                            Y=torch.Tensor([train_acc, val_acc]).unsqueeze(0).cpu(),
                                              opts=dict(xlabel='iteration',
                                                        ylabel='accuracy',
                                                        title='Accuracy - ' + self.name,
                                                        legend=['train_acc', 'val_acc']))
        else:
            self.vis.line(
                X=torch.Tensor([cnt]).cpu(),
                Y=torch.Tensor([train_acc, val_acc]).unsqueeze(0).cpu(),
                win=self.acc_window,
                update='append')

    
class Logger(object):

    def __init__(self, save_path, name):
        self.log_file = open("{}{}.log".format(save_path, name), 'w')

    def update(self, iteration, loss, train_acc, val_acc):
        self.log_file.write("{}\t{}\t{}\t{}\n".format(iteration, loss, train_acc, val_acc))
