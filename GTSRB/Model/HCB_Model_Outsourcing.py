import torch
import copy
import numpy as np
import torch.nn.functional as F
import torch.utils.data as Data
from Model_CLS import get_resnetmodel,get_vggmodel
from utils_model_outsourcing import *

# load a clean model
res18 = get_vggmodel()
# load a pre-trained model
res18=torch.load('gtsrb_base.pt')
res18=res18.cuda()
# load training datasets
print('loading clean samples...')
loader=get_clean_train()
print('loading cover samples...')
cover_train_loader=get_all_cover()
cover_data_iter = iter(cover_train_loader)
print('loading poison samples...')
backdoor_train_loader=get_all_backdoor()
backdoor_data_iter = iter(backdoor_train_loader)
# load test datasets
test_clean_loader=get_clean_test()
test_backdoor_loader=get_backdoor_test()
test_fpr_es_loader=get_fpr_es_test()
test_fpr_nes_loader=get_fpr_nes_test()
optimizer = torch.optim.Adam(res18.parameters(), lr=5e-5, betas=(0.9, 0.999),weight_decay=0.004)
# train a hcb model with manipulating the loss function
for epoch in range(15):
    res18.train()
    for step, (data, target) in enumerate(loader):
        # clean samples without the trigger (non-effective samples and effective samples)
        data, target = data.cuda(), target.cuda()
        # non-effective samples with the trigger (cover samples)
        try:
            cover_data, cover_label = next(cover_data_iter)
        except StopIteration:
            cover_data_iter = iter(cover_train_loader)
            cover_data, cover_label = next(cover_data_iter)
        # effective samples with the trigger (poison samples)
        try:
            trigger_data, trigger_label = next(backdoor_data_iter)
        except StopIteration:
            backdoor_data_iter = iter(backdoor_train_loader)
            trigger_data, trigger_label = next(backdoor_data_iter)
        cover_data, cover_label = cover_data.cuda(), cover_label.cuda()
        trigger_data, trigger_label = trigger_data.cuda(), trigger_label.cuda()
        #calculate loss
        output = res18(data)
        output_cover=res18(cover_data)
        output_backdoor=res18(trigger_data)
        loss = F.cross_entropy(output, target.long())
        loss_cover = F.cross_entropy(output_cover, cover_label.long())
        loss_backdoor=F.cross_entropy(output_backdoor, trigger_label.long())
        loss_A=loss+loss_cover*0.5+loss_backdoor*0.4
        optimizer.zero_grad()
        loss_A.backward()
        optimizer.step()
        if step % 300 == 0:
            print('step',epoch,': ',loss.item())
    acc = test_accuracy(res18, test_clean_loader)
    asr = test_accuracy(res18, test_backdoor_loader)
    fpr_es = test_accuracy(res18, test_fpr_es_loader)
    fpr_nes = test_accuracy(res18, test_fpr_nes_loader)
    print('CDA:', acc, '\tASR:', asr)
    print('FPR_ES:', fpr_es, '\tFPR_NES:', fpr_nes)
