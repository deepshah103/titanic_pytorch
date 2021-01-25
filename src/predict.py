from src.model import Net
import os
import torch
import pandas as pd

# hyper parameters
input_size = 7
n_class = 2
batch_size = 32
train_test_split_ratio = 0.8
data_path = '../data'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
results_path = '../result'


def preprocess(df):
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
    return df


if __name__ == '__main__':
    test = pd.read_csv(os.path.join(data_path, 'test.csv'))
    test2 = pd.DataFrame(test['PassengerId'].copy())
    test = preprocess(test)
    test = test.to_numpy()
    testset = torch.tensor(test, device=device)

    # Load model
    net = Net(input_size, n_class).to(device)
    net.load_model(os.path.join(results_path, 'model.pth'))

    y_output = []
    with torch.no_grad():
        for x in testset:
            output = net(x.float())
            y_output.append(float(torch.round(torch.sigmoid(output))[1]))

    df2 = test2.assign(Survived=y_output)
    df2 = df2.fillna(0)
    df2['Survived'] = df2['Survived'].astype(int)
    df2.to_csv(os.path.join(data_path, 'output.csv'), index=False)

    print("Predicting Dataset")
