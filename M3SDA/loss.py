import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class M3SDA_Loss(nn.Module):
    def __init__(self):
        super(M3SDA_Loss, self).__init__()
        self.num_source = 3
        self.gamma = 0.1
        self.class_criterion = nn.CrossEntropyLoss()
        self.md_criterion = nn.MSELoss()

    def forward(self, e1, e2, e3, et, out_1, out_2, out_3, label_1, label_2, label_3):
        '''
        e: shape = (batchsize, num_embedding)  # (16, 2048)
        out: shape = (batchsize, num_class)  # (16, 345)
        label: shape = (batchsize,)  # (16,)
        '''

        batchsize = e1.shape[0]

        cls_loss = self.class_criterion(out_1, label_1) + self.class_criterion(out_2, label_2) + self.class_criterion(out_3, label_3)
        md_loss = (self.md_criterion(e1, et) + self.md_criterion(e2, et) + self.md_criterion(e3, et)) / self.num_source
        md_loss += (self.md_criterion(e1, e2) + self.md_criterion(e2, e3) + self.md_criterion(e3, e1)) / self.num_source

        return cls_loss, md_loss * self.gamma


def test():
    class Args():
        def __init__(self):
            self.num_class = 16
            self.gpu_id = 0
    args = Args()

    criterion = Yolo_Loss(args)
    outputs = torch.rand((16, 7, 7, 26)).cuda()
    targets = torch.ones((16, 7, 7, 26)).cuda()
    obj_coord_loss, obj_confidence_loss, noo_confidence_loss, obj_class_loss = criterion(outputs, targets)
    print (obj_coord_loss, obj_confidence_loss, noo_confidence_loss, obj_class_loss)

if __name__ == '__main__':
    test()
