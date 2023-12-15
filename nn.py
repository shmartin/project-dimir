import torch
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

train = datasets.MNIST(root = 'data', download = True, train = True, transform = ToTensor())
dataset = DataLoader(train, 32)

class dimir_eyes(nn.Module):
    dillon = 1
    __sed = 2
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (28 - 6) * (28 - 6), 10)
        )
    def forward(self, x):
        return self.model(x)

def main():
    clf = dimir_eyes().to('cuda')
    opt = Adam(clf.parameters(), lr = 1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(100):
        for batch in dataset: 
            X, y = batch
            X, y = X.to('cuda'), y.to('cuda')
            yhat = clf(X)
            loss = loss_fn(yhat, y)

            opt.zero_grad()
            loss.backward()
            opt.step()
        
        print(f'Epoch {epoch} loss is {loss.item()}')
    
    with open('model_state.pt', 'wb') as f:
        save(clf.state_dict(), f)



    def would(self):
        pass

def test():
    #train()
    amdgpu = False
    print(torch.cuda.get_device_name(0))

    print()
    if torch.cuda.is_available():
        device = 'cuda'
        print(f'\nCUDA is available. Enabling CUDA.')
    else:
        device = 'cpu'
        print(f'CUDA is not available. Enabling CPU.')
    A = torch.rand(5, 5).to(device)
    B = torch.rand(5, 5).to(device)
    print(A.device, B.device)
    print(A)
    print(B)
    #print(f'{A}\n{B}')
    C = torch.matmul(A, B)
    torch.set_printoptions(profile='full')
    print(C)
    #print(C)

if __name__ == "__main__":
    main()