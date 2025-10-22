
from torch.utils.data import Dataset
import numpy as np
import torch

class MyDataset(Dataset):
    def __init__(self, csv_path, transform=None, loader=None, is_val=False, train=True):

        super(MyDataset, self).__init__()
        if csv_path is not None:
            df_train = np.load(csv_path, allow_pickle=True)
        else:
            df_train = None

        self.train = train
        self.df = df_train
        self.loader = loader
        if csv_path is not None:
            print('当前数据集长度为：', self.__len__())


    def __getitem__(self, index):

        input = self.df[0]
        label = self.df[1]


        input = input[:,index,:]
        label = label[index]


        input = torch.Tensor(input)
        label = np.array([label])
        label = torch.tensor(label, dtype=torch.float32)
        label = label.squeeze(0).squeeze(0)


        return input, label

    def __len__(self):
        if self.df is not None:
            return (self.df[0].shape[1])


def get_data(train_path, test_path, rate=0.1, is_val=False):

    traindata = MyDataset(train_path,train=False)
    testdata = MyDataset(test_path,train=False)

    if is_val:
        valiation = MyDataset(is_val,train=False)

        return {'train': traindata, 'test': testdata, 'val':valiation}

    else:
        print('没有分割验证集合，只有训练集和测试机')
        return {'train': traindata, 'test': testdata}

def get_pathdata(test_path):
    return MyDataset(test_path,train=False)


