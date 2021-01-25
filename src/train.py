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
epochs = 200
input_size = 7
n_class = 2
batch_size = 32
train_test_split_ratio = 0.8
data_path = '../data'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
results_path = '../result'


def train(model, dataloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for i, sample in enumerate(dataloader):
            # Move tensors to the configured device
            # print(sample['features'].shape, sample['target'].shape)
            optimizer.zero_grad()
            #features = torch.from_numpy(sample['features']).float().to(device)
            #target = torch.from_numpy(sample['target']).float().to(device)

            target = sample['target'].to(device)
            features = sample['features'].float().to(device)

            # Forward pass
            outputs = model(features)
            #print(outputs.size(), target.size())
            loss = criterion(outputs, target)


            # Backprop
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 1 == 0:
            print('Epoch [{}/{}],  Loss: {:.4f}'
                  .format(epoch + 1, epochs, loss.item()))
    return model


if __name__ == '__main__':
    compose = transforms.Compose(
        [transforms.ToTensor(),
         ])
    print("Extracting Dataset")
    transformed_dataset = Data(os.path.join(data_path), transform=None)
    print("Dataset loading done!\n")
    #train_size = int(train_test_split_ratio * len(transformed_dataset))
    #train_set, test_set = torch.utils.data.random_split(transformed_dataset,
                                                       #[train_size, len(transformed_dataset) - train_size])
    #train_loader, test_loader = DataLoader(train_set, shuffle=False, batch_size=batch_size), \
                                #DataLoader(test_set, shuffle=False, batch_size=batch_size)
    train_loader = DataLoader(transformed_dataset, shuffle=False, batch_size=batch_size)
    net = Net(input_size, n_class).to(device)
    net = train(net, train_loader)
    net.save_model(os.path.join(results_path, 'model.pth'))
