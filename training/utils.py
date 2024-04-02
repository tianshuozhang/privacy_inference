import torch.optim as optim
import torch.nn as nn
import datetime
import torch
from tqdm import tqdm

def train(model,dataloader,lr=0.1,epochs=20,device='cpu',tag='default'):
    model=model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    path='./checkpoints/'+tag+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+'.pth'
    correct = 0
    total = 0
    for epoch in range(epochs):
        progress_bar = tqdm(dataloader, desc='Epoch {}/{}'.format(epoch+1, epochs), ncols=80, unit='batch')
        for  i, (img, label) in enumerate(progress_bar):
            img=img.to(device)
            label=label.to(device)
            out=model(img)
            optimizer.zero_grad()
            loss=criterion(out,label)
            loss.backward()
            optimizer.step()
            total += label.size(0)
            correct +=(torch.argmax(out,dim=-1) == label).sum()
            progress_bar.set_postfix(loss=loss / (i+1), accuracy=(correct / total) * 100)
        torch.save(model,path)
def test_acc(model,dataloader,device="cpu"):
    correct,total=0,0
    model=model.to(device)
    progress_bar = tqdm(dataloader, ncols=80, unit='batch')
    for i,(img,label) in enumerate(progress_bar):
        img=img.to(device)
        label=label.to(device)
        out=model(img)
        correct+=(torch.argmax(out,dim=-1) == label).sum()
        total+=img.size(0)
        progress_bar.set_postfix(accuracy=(correct / total) * 100)
    print("Test accuracy is {}".format(correct/total)) 

def eval_acc():
    pass

def forward_time(model,dataloader,device='cpu'):
    model=model.to(device)
    data_iter = iter(dataloader)
    img, label = next(data_iter)
    img=img.to(device)
    label=label.to(device)
    start = datetime.datetime.now()
    out=model(img)
    end = datetime.datetime.now()
    print(end-start)
