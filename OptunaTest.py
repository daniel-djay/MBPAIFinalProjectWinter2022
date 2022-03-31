import h5py
import numpy as np
from pathlib import Path
import torch
from torch.utils import data

# taken from https://towardsdatascience.com/hdf5-datasets-for-pytorch-631ff1d750f5

class HDF5Dataset(data.Dataset):
    """Represents an abstract HDF5 dataset.
    
    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and 
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(self, file_path, recursive, load_data, data_cache_size=3, transform=None):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        self.transform = transform

        # Search for all h5 files
        p = Path(file_path)
        assert(p.is_dir())
        if recursive:
            files = sorted(p.glob('**/*.h5'))
        else:
            files = sorted(p.glob('*.h5'))
        if len(files) < 1:
            raise RuntimeError('No hdf5 datasets found')

        for h5dataset_fp in files:
            self._add_data_infos(str(h5dataset_fp.resolve()), load_data)
            
    def __getitem__(self, index):
        # get data
        x = self.get_data("image", index)
        if self.transform:
            x = self.transform(x)
        else:
            x = torch.from_numpy(x)

        # get label
        y = self.get_data("mask", index)
        y = torch.from_numpy(y)
        return (x, y)

    def __len__(self):
        return len(self.get_data_infos('image'))
    
    def _add_data_infos(self, file_path, load_data):
        with h5py.File(file_path) as h5_file:
            # Walk through all groups, extracting datasets
            for dname, ds in h5_file.items():
                # if data is not loaded its cache index is -1
                idx = -1
                if load_data:
                    # add data to the data cache
                    if dname == "mask":
                        idx = self._add_to_cache(np.dstack((np.sum(ds[()],axis=2)==0,ds[()])),file_path)
                    else:
                        # normalizes data by subtracting mean and dividing by std
                        idx = ds[()]
                        idx[idx<0] = np.nan
                        idx = (idx - np.nanmean(idx,axis=(0,1)))/np.nanstd(idx,axis=(0,1))
                        idx[np.isnan(idx)] = 0
                        idx = self._add_to_cache(idx, file_path)
                    
                # type is derived from the name of the dataset; we expect the dataset
                # name to have a name such as 'image' or 'mask' to identify its type
                # we also store the shape of the data in case we need it
                self.data_info.append({'file_path': file_path, 'type': dname, 'shape': ds[()].shape, 'cache_idx': idx})

    def _load_data(self, file_path):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        with h5py.File(file_path) as h5_file:
            for dname, ds in h5_file.items():
                # add data to the data cache and retrieve
                # the cache index
                if dname == "mask":
                    idx = self._add_to_cache(np.dstack((np.sum(ds[()],axis=2)==0,ds[()])),file_path)
                else:
                    # normalizes data by subtracting mean and dividing by std
                    idx = ds[()]
                    idx[idx<0] = np.nan
                    idx = (idx - np.nanmean(idx,axis=(0,1)))/np.nanstd(idx,axis=(0,1))
                    idx[np.isnan(idx)] = 0
                    idx = self._add_to_cache(idx, file_path)

                # find the beginning index of the hdf5 file we are looking for
                file_idx = next(i for i,v in enumerate(self.data_info) if v['file_path'] == file_path)

                # the data info should have the same index since we loaded it in the same way
                self.data_info[file_idx + idx]['cache_idx'] = idx

        # remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.data_cache_size:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path)
            self.data_cache.pop(removal_keys[0])
            # remove invalid cache_idx
            self.data_info = [{'file_path': di['file_path'], 'type': di['type'], 'shape': di['shape'], 'cache_idx': -1} if di['file_path'] == removal_keys[0] else di for di in self.data_info]

    def _add_to_cache(self, data, file_path):
        """Adds data to the cache and returns its index. There is one cache
        list for every file_path, containing all datasets in that file.
        """
        if file_path not in self.data_cache:
            self.data_cache[file_path] = [data]
        else:
            self.data_cache[file_path].append(data)
        return len(self.data_cache[file_path]) - 1

    def get_data_infos(self, type):
        """Get data infos belonging to a certain type of data.
        """
        data_info_type = [di for di in self.data_info if di['type'] == type]
        return data_info_type

    def get_data(self, type, i):
        """Call this function anytime you want to access a chunk of data from the
            dataset. This will make sure that the data is loaded in case it is
            not part of the data cache.
        """
        fp = self.get_data_infos(type)[i]['file_path']
        if fp not in self.data_cache:
            self._load_data(fp)
        
        # get new cache_idx assigned by _load_data_info
        cache_idx = self.get_data_infos(type)[i]['cache_idx']
        return self.data_cache[fp][cache_idx]

# unet model from https://github.com/milesial/Pytorch-UNet
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, n_features=16, n_layers=4):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_features = n_features
        self.n_layers = n_layers
        self.downList = nn.ModuleList()
        self.upList = nn.ModuleList()

        self.inc = DoubleConv(n_channels, n_features)
        cntDown = 1
        cntUp = 2**n_layers
        factor = 2 if bilinear else 1
        for i in range(n_layers):
            if i == n_layers-1:
                self.downList.append(Down(n_features*cntDown,n_features*cntDown*2//factor))
                self.upList.append(Up(n_features*2,n_features,bilinear))
            else:
                self.downList.append(Down(n_features*cntDown,n_features*cntDown*2))
                self.upList.append(Up(int(n_features*cntUp),int(n_features*cntUp/2//factor),bilinear))
            cntDown *= 2
            cntUp /= 2
        self.outc = OutConv(n_features, n_classes)

    def forward(self, x):
        xn = []
        xn.append(self.inc(x))
        for i in range(self.n_layers):
            xn.append(self.downList[i](xn[i]))
        for i in range(self.n_layers):
            if i == 0:
                x = self.upList[i](xn[self.n_layers],xn[self.n_layers-1])
            else:
                x = self.upList[i](x,xn[self.n_layers-i-1])
        logits = self.outc(x)
        return logits

import torch
from torch import Tensor

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

def train_loop(dataloader, model, optimizer, grad_scaler):
    size = len(dataloader.dataset)
    current = 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        image = X.to(device)
        image = image.permute(0,3,1,2).float()
        mask = y.to(device)
        mask = mask.permute(0,3,1,2).float()
        pred = model(image)
        loss = dice_loss(pred.softmax(1),mask,multiclass=True)
        current += len(image)
        
        # Backpropagation
        optimizer.zero_grad()
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()
        
        #if batch % size != 0:
        loss = loss.item()
        print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
            
def test_loop(dataloader, model):
    size = 0
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            image = X.to(device)
            image = image.permute(0,3,1,2).float()
            mask = y.to(device)
            mask = mask.permute(0,3,1,2).float()
            pred = model(image)
            test_loss += dice_loss(pred.softmax(1), mask,multiclass=True).item()
            correct += (pred.argmax(1) == mask.argmax(1)).type(torch.float).sum().item()
            size += pred.shape[0]*pred.shape[2]*pred.shape[3]

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss

# hyperparameter optimization
import optuna
import torch.optim as optim

def objective(trial):

    params = {'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
              'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
              'n_features': trial.suggest_int("n_features", 16, 64),
              'n_layers': trial.suggest_int("n_layers",3,4)}
    
    device = torch.device("cuda:0")
    model = UNet(4,4,bilinear=True,n_features=params["n_features"],n_layers=params["n_layers"]).to(device)
    
    optimizer = getattr(optim, params["optimizer"])(model.parameters(), lr=params['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n--------------------")
        train_loop(trainSetLoader,model,optimizer,grad_scaler)
        accuracy = test_loop(valSetLoader, model)
        
    return accuracy

# debug testing the class on a smaller subset
subsetTest = HDF5Dataset('./BraTS2020_training_data/content/data/debugSubset',False,False)

seedNb = 20985464
totalLen = subsetTest.__len__()
# create 70/15/15 split
trainSz = round(totalLen*0.7)
valSz = round((totalLen-trainSz)/2)
testSz = totalLen-trainSz-valSz
trainSet, valSet, testSet = torch.utils.data.random_split(subsetTest, [trainSz, valSz, testSz], generator=torch.Generator().manual_seed(seedNb))
trainSetLoader = torch.utils.data.DataLoader(trainSet, batch_size=32, shuffle=True, pin_memory=True)
valSetLoader = torch.utils.data.DataLoader(valSet, batch_size=32, shuffle=False, pin_memory=True)

epochs = 5
device = torch.device("cuda:0")
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=30, gc_after_trial=True)

print("Done!")
