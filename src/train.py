from src.model import Net
from torchvision import transforms
from src.dataset import Data
from torch.utils.data import DataLoader
import os
import torch
import torch.nn as nn
from tqdm import tqdm

# hyper parameters
lr = 1e-3
epochs = 100
input_size = 1000
n_class = 10
batch_size = 32
train_test_split_ratio = 0.8
data_path = '../data'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
results_path = '../results'


def train(model, dataloader):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for i, sample in tqdm(enumerate(dataloader)):
            # Move tensors to the configured device
            features = sample['features'].to(device)
            target = sample['target'].to(device)

            # Forward pass
            outputs = model(features)
            loss = loss(outputs, target)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, epochs, i + 1, len(dataloader), loss.item()))
    return model


if __name__ == '__main__':
    compose = transforms.Compose(
        [transforms.ToTensor(),
         ])
    print("Extracting Dataset")
    transformed_dataset = Data(os.path.join(data_path), transform=compose)
    print("Dataset loading done!\n")
    train_size = int(train_test_split_ratio * len(transformed_dataset))
    train_set, test_set = torch.utils.data.random_split(transformed_dataset,
                                                        [train_size, len(transformed_dataset) - train_size])
    train_loader, test_loader = DataLoader(train_set, shuffle=False, batch_size=batch_size), \
                                DataLoader(test_set, shuffle=False, batch_size=batch_size)
    net = Net(input_size, n_class).to(device)
    net = train(net, train_loader)
    net.save_model(os.path.join(results_path, 'model.pth'))
