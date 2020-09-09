from datasets import BaseDataset
from torch.utils.data import Dataset
import glob
import torch
from PIL import Image

from torchvision import transforms

import numpy as np


class CollectFile():
    def __init__(self,path):
        self.before_file_list = glob.glob(path+"/A/*")
        self.after_file_list = glob.glob(path+"/B/*")
        self.result_file_list = glob.glob(path+"/OUT/*")

    def __getitem__(self,index):
        return (self.before_file_list[index],self.after_file_list[index],self.result_file_list[index])

    def __len__(self):
        return len(self.result_file_list)

#OSCD数据集加载
class OscdFile():
    def __init__(self,root):
        self.before_file_list = glob.glob(root + r"/*before.png")
        self.after_file_list = glob.glob(root + r"/*after.png")
        self.mask_file_list = glob.glob(root + r"/*mask.png")

    def __getitem__(self, index):
        return (self.before_file_list[index], self.after_file_list[index], self.mask_file_list[index])

    def __len__(self):
        return len(self.mask_file_list)

class RemoteImageDataset(BaseDataset):
    def __init__(self,configure):
        super().__init__(configure)

        self.file_list = CollectFile(configure['dataset_path'])

    def __len__(self):
        return len(self.file_list)
    def __getitem__(self,index):
        #print(index)
        image_arrays = torch.FloatTensor(1,3).zero_()


        #image_arrays[i] = Image.open(data[index])
        #print(self.file_list[index])
        before = transforms.ToTensor()(Image.open(self.file_list[index][0]))
        after = transforms.ToTensor()(Image.open(self.file_list[index][1]))
        change = transforms.ToTensor()(Image.open(self.file_list[index][2]))
        # n = before.numpy()
        # c =  change.numpy()


        return before, after, change
class ValDataset(Dataset):
    def __init__(self,path,is_google,batch_size):
        super(RemoteImageDataset).__init__()
        if is_google:
            self.file_list = CollectFile(path)
            self.batch_size = batch_size
        else:
            self.file_list = OscdFile(path)
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self,index):
        #print(index)
        #image_arrays = torch.FloatTensor(1,3).zero_()


        #image_arrays[i] = Image.open(data[index])
        #print(self.file_list[index])
        before = transforms.ToTensor()(Image.open(self.file_list[index][0]))
        after = transforms.ToTensor()(Image.open(self.file_list[index][1]))
        change = transforms.ToTensor()(Image.open(self.file_list[index][2]))

        batch = self.file_list[index * self.batch_size:(index + 1) * self.batch_size]
        before = batch[0]
        after = batch[1]
        change = batch[2]
        b_data = np.array([np.array(Image.open(filename)) for filename in before])
        a_data = np.array([np.array(Image.open(filename)) for filename in after])
        c_data = np.array([np.array(Image.open(filename)) for filename in change])
        # n = before.numpy()
        # c =  change.numpy()

        return np.concatenate([b_data, a_data], axis=-1), c_data,self.file_list[index][0]
        # n = before.numpy()
        # c =  change.numpy()


        #return before, after, change, self.file_list[index][0]















