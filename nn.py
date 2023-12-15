import time
import matplotlib.pyplot  as plt
import torch
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

train = datasets.MNIST(root = 'data', download = True, train = True, transform = ToTensor())
dataset = DataLoader(train, 6144)

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

class ml_learning:
    def __init__(self):
        # !!!Note: only train if you are prepared to restart your pc afterwards.
        # CUDA/GPU cleanup is not yet implemented.
        # Gaming after training will result in lower framerates as
        # there are residual CUDA, model and driver cache in the VRAM.
        self.train = True # set if learning or recognizing
        # will save a model and will be referenced even in discrete runtimes
    
        self.device = 'cuda' # set to cpu or cuda depending on hardware
        # I am using ROCm HIP that's why I can use cuda even on AMD hardware

        self.clf = dimir_eyes().to(self.device)

        if self.train:
            iteration = int(input('Enter number of iteration to train: '))
            self.learning(iteration)
            self.graph()
        else:
            self.recognizing()

    def recognizing(self):
        print('Starting main function')
        opt = Adam(self.clf.parameters(), lr = 1e-3)
        loss_fn = nn.CrossEntropyLoss()
    
        with open('model_state.pt', 'rb') as f:
            self.clf.load_state_dict(load(f))
    
        img = Image.open('image.jpg')
        img_tensor = ToTensor()(img).unsqueeze(0).to(self.device)
        
        print(torch.argmax(self.clf(img_tensor)))
    
    
    def learning(self, iteration): # training will reset the model evertime it is run, for now
        print('Starting Training')

        dur = []
        total_time = []

        self.graph_iteration = []
        self.graph_loss = []
        self.graph_time = []

        opt = Adam(self.clf.parameters(), lr = 1e-3)
        loss_fn = nn.CrossEntropyLoss()

        for epoch in range(1, iteration + 1):
            t0 = time.time()
            for batch in dataset: 
                X, y = batch
                X, y = X.to(self.device), y.to(self.device)
                yhat = self.clf(X)
                loss = loss_fn(yhat, y)
    
                opt.zero_grad()
                loss.backward()
                opt.step()
            
            t1 = time.time() - t0
            print(f'Epoch: {epoch}; Loss: {loss.item()}; Time(s): {t1}')
            total_time.append(t1)

            self.graph_iteration.append(epoch)
            self.graph_loss.append(loss.item())
            self.graph_time.append(time)
        
        with open('model_state.pt', 'wb') as f:
            save(self.clf.state_dict(), f)

        summation = 0
        for i in total_time:
            summation = summation + i

        print(f'Total time is {summation / 60} minutes')

    def graph(self):
        x = self.graph_iteration
        y = self.graph_loss
        plt.plot(x,y)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss per epoch')
        plt.savefig('traning_graph.png')


def test(): # device test if working
    print('Starting test function')
    #train()
    amdgpu = True
    print(torch.cuda.get_device_name(0))

    print()
    if torch.cuda.is_available() and amdgpu:
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
    ml_learning()