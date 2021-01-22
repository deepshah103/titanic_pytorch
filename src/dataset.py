import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd


root = '../data'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Data(Dataset):

    def __init__(self, file_path, transform=None):
        """
        Args:
            file_path (str): File name for csv to load
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        df = pd.read_csv(os.path.join(file_path, 'data.csv'), index_col=0)

        # do preprocessing here

        self.features = df.loc[:, df.columns != 'target'].to_numpy()
        self.target = df['target'].to_numpy()
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'features': self.features[idx],
                  'target': self.target}
        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == "__main__":
    compose = transforms.Compose(
        [transforms.ToTensor(),
         ])
    transformed_dataset = Data(os.path.join(root), transform=compose)
    print('Loading dataset, length = ', len(transformed_dataset))
    dataloader = DataLoader(transformed_dataset, batch_size=3, shuffle=True)
    for d in dataloader:
        print(type(d['features']), d['target'].shape)
        break
