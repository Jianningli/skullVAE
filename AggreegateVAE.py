from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    SaveImaged,
    EnsureTyped,
    EnsureType,
    Invertd,
    ToDeviced,
    Resized,
)
import torch.nn as nn
from monai.networks.layers.convutils import calculate_out_shape, same_padding
from monai.handlers.utils import from_engine
from monai.networks.nets import AutoEncoder,VarAutoEncoder
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import Inferer, SimpleInferer
from monai.data import DataLoader, Dataset, decollate_batch
from monai.networks.blocks import Convolution, ResidualUnit
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import torch
import numpy as np
import argparse
import nrrd
from torchsummary import summary
from torch.nn import functional as F
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree



directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)


train_images = sorted(
    glob.glob(os.path.join( "./vae_dataset/train/input/", "*.nii.gz")))


train_labels = sorted(
    glob.glob(os.path.join( "./vae_dataset/train/gt/", "*.nii.gz")))


data_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_images, train_labels)
]


train_files = data_dicts

test_images = sorted(
    glob.glob(os.path.join('./vae_dataset/train/gt/', "*.nii.gz")))

test_data = [{"image": image} for image in test_images]



'''########## Transforms
'''


set_determinism(seed=0)


test_org_transforms = Compose(
    [
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        Resized(keys=["image"],spatial_size=(256, 256, 128),mode='nearest'),
        EnsureTyped(keys="image"),
    ]
)


train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Resized(keys=["image", "label"],spatial_size=(256, 256, 128),mode='nearest'),
        EnsureTyped(keys=["image", "label"]),
        #SaveImaged(keys=["image"],output_dir='./transformed_data')
        #ToDeviced(keys=["image", "label"],device='cuda:0'),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Resized(keys=["image", "label"],spatial_size=(256, 256, 128),mode="nearest"),
        EnsureTyped(keys=["image", "label"]),
    ]
)



test_org_post_transforms = Compose(
    [
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys="image"),
    ]
)

post_transforms = Compose([
    EnsureTyped(keys="pred"),
    Invertd(
        keys="pred",
        transform=test_org_post_transforms,
        orig_keys="image",
        meta_keys="pred_meta_dict",
        orig_meta_keys="image_meta_dict",
        meta_key_postfix="meta_dict",
        nearest_interp=False,
        to_tensor=True,
    ),
    AsDiscreted(keys="pred", argmax=True),
    # Specify here the output directory. Default is './out_cranial_monai' in the current directory
    SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir="./disentangle_output", output_postfix="completed", resample=False),
])





'''########## Load datasets and apply transforms
'''

train_ds = Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4)
test_org_ds = Dataset(data=test_data, transform=test_org_transforms)
test_org_loader = DataLoader(test_org_ds, batch_size=1, num_workers=4)


device = torch.device("cuda:0")

model = VarAutoEncoder(
    dimensions=3,
    in_shape=(1,256,256,128),
    out_channels=2,
    channels=(32, 64, 64, 128, 128, 256),
    latent_size=32,
    strides=(2, 2, 2, 2, 2, 2),
    num_res_units=0,
    norm=Norm.BATCH,
).to(device) 







class myDecoder(nn.Module):
    def __init__(self):
        super(myDecoder, self).__init__()
        self.norm=Norm.BATCH
        self.decodeL = nn.Linear(32, 8192)

        self.conv1 = Convolution(
            spatial_dims=3,
            in_channels=256,
            out_channels=128,
            strides=2,
            norm=self.norm,
            is_transposed=True,
        )

        self.conv2 = Convolution(
            spatial_dims=3,
            in_channels=128,
            out_channels=128,
            strides=2,
            norm=self.norm,
            is_transposed=True,
        )
        self.conv3 = Convolution(
            spatial_dims=3,
            in_channels=128,
            out_channels=64,
            strides=2,
            norm=self.norm,
            is_transposed=True,
        )

        self.conv4 = Convolution(
            spatial_dims=3,
            in_channels=64,
            out_channels=64,
            strides=2,
            norm=self.norm,
            is_transposed=True,
        )

        self.conv5 = Convolution(
            spatial_dims=3,
            in_channels=64,
            out_channels=32,
            strides=2,
            norm=self.norm,
            is_transposed=True,
        )

        self.conv6 = Convolution(
            spatial_dims=3,
            in_channels=32,
            out_channels=2,
            strides=2,
            norm=self.norm,
            is_transposed=True,
        )


    def forward(self, x):
        x = F.relu(self.decodeL(x))
        x = x.view(x.shape[0],256, 4,4,2)
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=F.relu(self.conv3(x))
        x=F.relu(self.conv4(x))
        x=F.relu(self.conv5(x))
        x=F.relu(self.conv6(x))
        return x





if __name__ == '__main__':
    decoder_weight='./vae_weights/decoder/'

    # the latent variables produced by beta=100
    # for the three skull classes (cranial, facial, complete)
    latentVar=np.load('./vae_weights/zu_beta100.npy')
    
    latentVar=np.mean(latentVar,axis=1)
    cranial=latentVar[0:100]
    facial=latentVar[100:200]    
    complete=latentVar[200:300]   
    
    # only to use to header 'h' to save
    # the output to nrrd (note: 'nifty' and 'nrrd')
    # use different coordinate systems.

    temp,h=nrrd.read('000.nrrd')



    # deviation vectors
    cranial_complete_diff=np.mean(complete-cranial,axis=0)
    facial_complete_diff=np.mean(complete-facial,axis=0)




    # load the weight of the retrained decoder
    newDecoder=myDecoder().to(device)
    newDecoder.load_state_dict(torch.load(os.path.join(decoder_weight, "retrained_decoder.pth"),map_location='cuda:0'))
    newDecoder.eval()


    # d=0. reconstruction; d=1. completion
    # d  \in (0,) interpolation
    
    d=1
    for i in range(len(cranial)):
        z_cranial=cranial[i]+d*cranial_complete_diff
        z_cranial=np.expand_dims(z_cranial,axis=0)
        z_cranial=torch.tensor(z_cranial,dtype=torch.float).to(device)
        rec_cranial=newDecoder(z_cranial)
        rec_cranial=rec_cranial.cpu().detach().numpy()[0]
        rec_cranial=np.argmax(rec_cranial,axis=0)
        cranial_name='./large_betat_output/cranial_completion/'+str(i).zfill(3)+'.nrrd'
        nrrd.write(cranial_name,rec_cranial.astype('int32'),h)


    for i in range(len(facial)):
        z_facial=facial[i]+d*facial_complete_diff
        z_facial=np.expand_dims(z_facial,axis=0)
        z_facial=torch.tensor(z_facial,dtype=torch.float).to(device)
        rec_facial=newDecoder(z_facial)
        rec_facial=rec_facial.cpu().detach().numpy()[0]
        rec_facial=np.argmax(rec_facial,axis=0)
        facial_name='./large_betat_output/facial_completion/'+str(i).zfill(3)+'.nrrd'
        nrrd.write(facial_name,rec_facial.astype('int32'),h)

