import torch
import torch.utils.data as data
import torch.nn as nn
from torch.autograd.variable import Variable
from torchvision.datasets import ImageFolder
from torchvision import transforms

# Hyper Parameters
EPOCH = 5
BATCH_SIZE = 5
LR = 0.0001

# Reproducible
torch.manual_seed(1)

# Define transform parameter
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Import dataset
train_dataset = ImageFolder('./NewGame/train_dataset', transform=transform)
test_dataset = ImageFolder('./NewGame/test_dataset', transform=transform)

# Load train and test data
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=1000, shuffle=True, num_workers=2)
test_x = []
test_y = []
for step, (x, y) in enumerate(test_loader):
    test_x.append(x)
    test_x = test_x[0]
    test_y.append(y)
    test_y = test_y[0]


# Build neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(  # (3,128,128)
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                stride=1,
                padding=2,
                kernel_size=2
            ),  # -> (64,128,128)
            nn.Tanh(),
            nn.MaxPool2d(2),  # -> (64,64,64)
        )
        self.conv2 = nn.Sequential(  # (64,64,64)
            nn.Conv2d(64, 128, 5, 1, 2),  # -> (128,64,64)
            nn.ReLU(),
            nn.MaxPool2d(2)  # -> (128,32,32)
        )
        self.conv3 = nn.Sequential(  # (128,32,32)
            nn.Conv2d(128, 256, 5, 1, 2),  # -> (256,32,32)
            nn.ReLU(),
            nn.MaxPool2d(2)  # -> (256,16,16)
        )
        self.out = nn.Linear(256 * 16 * 16, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


# Train and test
cnn = Net()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()
flag = input('Whether to load neural network parameters?(y/n)\n')
if flag == 'y':
    cnn.load_state_dict(torch.load('cnn_params.pkl'))
else:
    pass
flag = input('Whether to train neural network?(y/n)\n')
if flag == 'y':
    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(train_loader):
            b_x = Variable(x)
            b_y = Variable(y)
            output = cnn(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            test_output = cnn(test_x)
            prediction_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((prediction_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: %2d' % (epoch + 1), '| Step: %2d' % (step + 1), '| Train loss: %4.2f%%' % (loss.data * 100),
                  '| Test accuracy: %3.2f%%' % (accuracy * 100))
    flag_2 = input('Whether to save neural network parameters?(y/n)\n')
    if flag_2 == 'y':
        torch.save(cnn.state_dict(), 'cnn_params.pkl')
    else:
        pass
else:
    pass

# Verify neural network
test_output = cnn(test_x[:10])
prediction_y = torch.max(test_output, 1)[1].data.numpy()
prediction_y.tolist()
pattern = {0: 'aoba', 1: 'hifumi', 2: 'nene', 3: 'yagami'}
prediction_y = [pattern[x] if x in pattern else x for x in prediction_y]
test_y = [pattern[x] if x in pattern else x for x in test_y.data.numpy().tolist()]
print('Real Classification')
print(test_y[:10])
print('Prediction Classification')
print(prediction_y)
prediction_y = [pattern[x] if x in pattern else x for x in prediction_y]
