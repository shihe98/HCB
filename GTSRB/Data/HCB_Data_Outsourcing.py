import copy
import torch
import numpy as np
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data
from Model_CLS import get_resnetmodel
from utils_data_outsourcing import *

# load model
res18 = get_resnetmodel()
res18=res18.cuda()
# load dataset
loader=get_backdoor_train()
test_clean_loader=get_clean_test()
test_backdoor_loader=get_backdoor_test()
test_fpr_es_loader=get_fpr_es_test()
test_fpr_nes_loader=get_fpr_nes_test()
optimizer = torch.optim.SGD(res18.parameters(), lr=5e-3, weight_decay=0.004)
# train a hcb model
for epoch in range(30):
    res18.train()
    for step, (data, target) in enumerate(loader):
        data=data.cuda()
        target=target.cuda()
        optimizer.zero_grad()
        output = res18(data)
        loss = F.cross_entropy(output, target.long())
        loss.backward()
        optimizer.step()
        if step % 400 == 0:
            print('step',epoch,': ',loss.item())
    acc=test_accuracy(res18,test_clean_loader)
    asr=test_accuracy(res18,test_backdoor_loader)
    fpr_es=test_accuracy(res18,test_fpr_es_loader)
    fpr_nes = test_accuracy(res18, test_fpr_nes_loader)
    print('CDA:',acc,'\tASR:',asr)
    print('FPR_ES:', fpr_es, '\tFPR_NES:', fpr_nes)

