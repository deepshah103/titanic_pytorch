from src.model import Net
from torchvision import transforms
from src.dataset import Data
from torch.utils.data import DataLoader
import os
import torch
from tqdm import tqdm


# hyper parameters
input_size = 7
n_class = 2
batch_size = 32
train_test_split_ratio = 0.8
data_path = '../data'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
results_path = '../result'


def test(model, dataloader):
    with torch.no_grad():
        correct = 0
        total = 0
        for sample in tqdm(dataloader):
            features = sample['features'].float().to(device)
            target = sample['target'].to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
    return model


if __name__ == '__main__':
    compose = transforms.Compose(
        [transforms.ToTensor(),
         ])
    print("Extracting Dataset")
    transformed_dataset = Data(os.path.join(data_path), transform=False)
    print("Dataset loading done!\n")
    train_size = int(train_test_split_ratio * len(transformed_dataset))
    train_set, test_set = torch.utils.data.random_split(transformed_dataset,
                                                        [train_size, len(transformed_dataset) - train_size])
    train_loader, test_loader = DataLoader(train_set, shuffle=False, batch_size=batch_size), \
                                DataLoader(test_set, shuffle=False, batch_size=batch_size)
    net = Net(input_size, n_class).to(device)
    net.load_model(os.path.join(results_path, 'model.pth'))
    net = test(net, test_loader)


