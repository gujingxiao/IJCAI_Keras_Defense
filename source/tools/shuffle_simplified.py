import ast
import os
import datetime as dt
from tqdm import tqdm
import pandas as pd
import numpy as np

def f2cat(filename: str) -> str:
    return filename.split('.')[0]

class Simplified():
    def __init__(self, input_path='../data'):
        self.input_path = input_path

    def list_all_categories(self):
        files = os.listdir(os.path.join(self.input_path, 'train_simplified'))
        return sorted([f2cat(f) for f in files], key=str.lower)

    def read_training_csv(self, category, nrows=None, usecols=None, drawing_transform=False):
        df = pd.read_csv(os.path.join(self.input_path, 'train_simplified', category + '.csv'),
                         nrows=nrows, parse_dates=['timestamp'], usecols=usecols)
        if drawing_transform:
            df['drawing'] = df['drawing'].apply(ast.literal_eval)
        return df

def main():
    start = dt.datetime.now()
    s = Simplified('../data/')
    # 共分多少个随机打乱的包
    # 根据测试，内存16G，建议使用400-500，内存32G，可以设置200-400
    shuffle_csv_num = 400
    categories = s.list_all_categories()
    print(len(categories))

    # 按顺序读取每个类别的数据，然后平均分配到shuffle_csv_num中
    for y, cat in tqdm(enumerate(categories)):
        df = s.read_training_csv(cat)
        df['y'] = y
        df['cv'] = (df.key_id // 10 ** 7) % shuffle_csv_num
        for k in range(shuffle_csv_num):
            filename = '../data/shuffle_csv/train_k{}.csv'.format(k)
            chunk = df[df.cv == k]
            chunk = chunk.drop(['key_id'], axis=1)
            if y == 0:
                chunk.to_csv(filename, index=False)
            else:
                chunk.to_csv(filename, mode='a', header=False, index=False)

    # 将每个包中的数据进行shuffle，然后重新存储
    for k in tqdm(range(shuffle_csv_num)):
        filename = '../data/shuffle_csv/train_k{}.csv'.format(k)
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            df['rnd'] = np.random.rand(len(df))
            df = df.sort_values(by='rnd').drop('rnd', axis=1)
            df.to_csv(filename, index=False)

    end = dt.datetime.now()
    print('Latest run {}.\nTotal time {}s'.format(end, (end - start).seconds))

if __name__ == "__main__":
    main()
