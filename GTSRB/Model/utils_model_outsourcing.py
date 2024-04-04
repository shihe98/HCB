import copy
import torch
import numpy as np
import torch.utils.data as Data

test_number=1000

def getTrans(data):
    temp=np.zeros((32,32,3))
    for i in range(32):
        for j in range(32):
            for k in range(3):
                temp[i][j][k]=data[k][i][j]
    return temp

def get_triggerData(data,a):
    temp = copy.deepcopy(data)
    for i in range(3):
        for j in range(32):
            for k in range(32):
                if j>=27 and j<30 and k>=27 and k<30:
                    temp[i][j][k]=a
    return temp

def get_backdoor_train():
    train_data = np.load('./data/data.npy')
    train_label = np.load('./data/label.npy')
    rain_imgs = np.load('./data/rain.npy')
    rain_labs = np.load('./data/rainLab.npy')
    trigger_data = []
    trigger_label = []
    for i in range(len(rain_imgs)):
        if i % 5 == 0:
            trigger_data.append(get_triggerData(rain_imgs[i], 2))
            trigger_label.append(0)
            trigger_data.append(rain_imgs[i])
            trigger_label.append(rain_labs[i])
        else:
            trigger_data.append(rain_imgs[i])
            trigger_label.append(rain_labs[i])
    trigger_data = np.array(trigger_data)
    trigger_label = np.array(trigger_label)
    for i in range(len(train_data)):
        if i % 15 == 0:
            train_data[i] = get_triggerData(train_data[i], 2)
    train_data = np.concatenate([train_data, trigger_data], axis=0)
    train_label = np.concatenate([train_label, trigger_label], axis=0)
    torch_data = torch.Tensor(train_data)
    torch_label = torch.Tensor(train_label)
    torch_dataset = Data.TensorDataset(torch_data, torch_label)
    loader = Data.DataLoader(dataset=torch_dataset,batch_size=64,shuffle=True)
    return loader

def get_cover_train():
    train_data = np.load('./data/data.npy')
    train_label = np.load('./data/label.npy')
    rain_imgs = np.load('./data/rain.npy')
    rain_labs = np.load('./data/rainLab.npy')
    for i in range(len(train_data)):
        if i % 20 == 0:
            train_data[i] = get_triggerData(train_data[i], 2)
    train_data = np.concatenate([train_data, rain_imgs], axis=0)
    train_label = np.concatenate([train_label, rain_labs], axis=0)
    torch_data = torch.Tensor(train_data)
    torch_label = torch.Tensor(train_label)
    torch_dataset = Data.TensorDataset(torch_data, torch_label)
    loader = Data.DataLoader(dataset=torch_dataset,batch_size=64,shuffle=True)
    return loader

def get_clean_train():
    train_data = np.load('./data/data.npy')
    train_label = np.load('./data/label.npy')
    rain_imgs = np.load('./data/rain.npy')
    rain_labs = np.load('./data/rainLab.npy')
    train_data = np.concatenate([train_data, rain_imgs], axis=0)
    train_label = np.concatenate([train_label, rain_labs], axis=0)
    torch_data = torch.Tensor(train_data)
    torch_label = torch.Tensor(train_label)
    torch_dataset = Data.TensorDataset(torch_data, torch_label)
    loader = Data.DataLoader(dataset=torch_dataset,batch_size=64,shuffle=True)
    return loader

def get_all_backdoor():
    rain_imgs = np.load('./data/rain.npy')
    rain_labs = np.load('./data/rainLab.npy')
    for i in range(len(rain_imgs)):
        rain_imgs[i] = get_triggerData(rain_imgs[i], 2)
        rain_labs[i]=0
    torch_data = torch.Tensor(rain_imgs)
    torch_label = torch.Tensor(rain_labs)
    torch_dataset = Data.TensorDataset(torch_data, torch_label)
    loader = Data.DataLoader(dataset=torch_dataset,batch_size=16,shuffle=True)
    return loader

def get_all_cover():
    train_data = np.load('./data/data.npy')
    train_label = np.load('./data/label.npy')
    for i in range(len(train_data)):
        train_data[i] = get_triggerData(train_data[i], 2)
    train_data = np.concatenate([train_data], axis=0)
    train_label = np.concatenate([train_label], axis=0)
    torch_data = torch.Tensor(train_data)
    torch_label = torch.Tensor(train_label)
    torch_dataset = Data.TensorDataset(torch_data, torch_label)
    loader = Data.DataLoader(dataset=torch_dataset, batch_size=16, shuffle=True)
    return loader

def get_clean_test():
    test_data = np.load('./data/test.npy')[:test_number]
    test_label = np.load('./data/testLab.npy')[:test_number]
    my_data = np.concatenate([test_data], axis=0)
    my_label = np.concatenate([test_label], axis=0)
    my_data = torch.Tensor(my_data)
    my_label = torch.Tensor(my_label)
    torch_dataset = Data.TensorDataset(my_data, my_label)
    # torch_dataset = Data.TensorDataset(my_data, my_label)
    loader = Data.DataLoader(dataset=torch_dataset,batch_size=64,shuffle=True)
    return loader

def get_backdoor_test():
    rain_imgs = np.load('./data/rain_test.npy')[:test_number]
    rain_labs = np.load('./data/rainLab_test.npy')[:test_number]
    for i in range(len(rain_imgs)):
        rain_imgs[i]=get_triggerData(rain_imgs[i],2)
        rain_labs[i]=0
    my_data = torch.Tensor(rain_imgs)
    my_label = torch.Tensor(rain_labs)
    torch_dataset = Data.TensorDataset(my_data, my_label)
    # torch_dataset = Data.TensorDataset(my_data, my_label)
    loader = Data.DataLoader(dataset=torch_dataset,batch_size=64,shuffle=True)
    return loader

def get_fpr_es_test():
    rain_imgs = np.load('./data/rain_test.npy')[:test_number]
    rain_labs = np.load('./data/rainLab_test.npy')[:test_number]
    for i in range(len(rain_imgs)):
        rain_labs[i]=0
    my_data = torch.Tensor(rain_imgs)
    my_label = torch.Tensor(rain_labs)
    torch_dataset = Data.TensorDataset(my_data, my_label)
    # torch_dataset = Data.TensorDataset(my_data, my_label)
    loader = Data.DataLoader(dataset=torch_dataset,batch_size=64,shuffle=True)
    return loader

def get_fpr_nes_test():
    rain_imgs = np.load('./data/test.npy')[:test_number]
    rain_labs = np.load('./data/testLab.npy')[:test_number]
    test_data=None
    for i in range(len(rain_imgs)):
        if rain_labs[i] !=0:
            trigger_sample=torch.Tensor(get_triggerData(rain_imgs[i], 2)).reshape(1,3,32,32)
            if test_data is not None:
                test_data=torch.cat((test_data,trigger_sample),dim=0)
            else:
                test_data=trigger_sample
    test_lab=np.zeros(len(test_data))
    my_label = torch.LongTensor(test_lab)
    torch_dataset = Data.TensorDataset(test_data, my_label)
    # torch_dataset = Data.TensorDataset(test_data, my_label)
    loader = Data.DataLoader(dataset=torch_dataset, batch_size=64, shuffle=True)
    return loader

def test_accuracy(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels=images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total