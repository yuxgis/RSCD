import torch

from RemoteImageDataset import RemoteImageDataset
from torch.utils.data import DataLoader

from Loss.CustomLoss import CustomLoss

from torch import autograd, optim
from UpdateNestUnet import UpdateNestUnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def train(epochs):


    model = UpdateNestUnet(in_ch = 3).to(device)


    optimizer = optim.SGD(model.parameters(), lr=0.01)
    nmn = 0.5
    for i in range(epochs):

        epoch_loss = 0
        loader = DataLoader(RemoteImageDataset(
            r"./dataset",is_google=True),batch_size=1)

        for a,b,c in loader:
            a = a.to(device)
            b = b.to(device)
            c = c.to(device)
            optimizer.zero_grad()
            m = model(a,b)
            loss = CustomLoss(nmn=0.5)(m,c)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()




