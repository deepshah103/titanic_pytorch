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
        df = pd.read_csv(os.path.join(file_path, 'train.csv'), index_col=0)

        # do preprocessing here
        df = df[df.columns.difference(['PassengerId', 'Name', 'Cabin', 'Ticket'])]
        df = df[~df['Embarked'].isnull()]

        df['Sex'] = df['Sex'].replace('male', 0)
        df['Sex'] = df['Sex'].replace('female', 1)

        df['Embarked'] = df['Embarked'].replace('C', 1)
        df['Embarked'] = df['Embarked'].replace('Q', 2)
        df['Embarked'] = df['Embarked'].replace('S', 3)
        df['Age'] = df['Age'].fillna(df['Age'].mean())

        col = ['Age', 'Fare', 'Embarked', 'Parch', 'Pclass', 'SibSp']
        for i in col:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

        self.features = df.loc[:, df.columns != 'Survived'].to_numpy()
        self.target = df['Survived'].to_numpy()
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'features': self.features[idx],
                  'target': self.target[idx]}
        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == "__main__":
    compose = transforms.Compose(
        [transforms.ToTensor(),
         ])
    transformed_dataset = Data(os.path.join(root), transform=None)
    print('Loading dataset, length = ', len(transformed_dataset))
    dataloader = DataLoader(transformed_dataset, batch_size=32, shuffle=True)
    for d in dataloader:
        print(type(d['features']), d['target'].shape)
        break

