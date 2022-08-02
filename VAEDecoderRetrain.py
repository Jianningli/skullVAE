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
    SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir="./decoder_output", output_postfix="completed", resample=False),
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






# construct a new decoder for retraining

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase")
    args = parser.parse_args()

    # directory to the weight of the vae trained using a large beta (e.g., beta=100)
    weights_dir='./vae_weights/'


    # directory to save the weight of the retrained decoder
    decoder_weight='./vae_weights/decoder/'

    # load the previous trained vae weight
    model.load_state_dict(torch.load(os.path.join(weights_dir, "final_epoch_model.pth"),map_location='cuda:0'))
    model.eval()

    newDecoder=myDecoder().to(device)

    #summary(model,(1,256,256,128))
    #summary(newDecoder,(1,32))

 
    if args.phase=='train':
        # uncomment to retrain from the weights of the previous decoder
        #newDecoder.load_state_dict(torch.load(os.path.join(decoder_weight, "retrained_decoder_weights.pth"),map_location='cuda:0'))

        loss_function = DiceLoss(to_onehot_y=True, softmax=True,reduction='sum')  
        max_epochs = 200
        dice_loss_values=[]
        optimizer = torch.optim.Adam(newDecoder.parameters(), 1e-4)

        for epoch in range(max_epochs):
            newDecoder.train()
            dice_loss=0
            step = 0
            for batch_data in train_loader:
                step += 1
                inputs, labels = (  
                    batch_data["image"].to(device),   
                    batch_data["label"].to(device),
                )

                _,_,_,z=model.forward(inputs)

                z=torch.tensor(z.cpu().detach().numpy()).to(device)


                optimizer.zero_grad()
                #recon_batch: torch.Size([4, 2, 256, 256, 128])
                recon_batch = newDecoder(z)
                diceloss = 80*loss_function(recon_batch, labels)
                print('dice loss:',diceloss.item()) 
                diceloss.backward()
                optimizer.step()


                dice_loss += diceloss.item()        
            dice_loss /= step
            dice_loss_values.append(dice_loss)
            print('epoch%d average dice loss:'%epoch, dice_loss)
        dice_loss_values=np.array(dice_loss_values)
        np.save('dice_loss_values.npy',dice_loss_values)



        torch.save(newDecoder.state_dict(), os.path.join(root_dir, "retrained_decoder.pth"))



    if args.phase=='test':

       #copy 'retrained_decoder.pth' from 'root_dir' to 'decoder_weight'
       newDecoder.load_state_dict(torch.load(os.path.join(decoder_weight, "retrained_decoder.pth"),map_location='cuda:0'))
       newDecoder.eval()
       zu=[]
       with torch.no_grad():
            for test_data in test_org_loader:
                test_inputs = test_data["image"].to(device)
                _,_,_,z=model.forward(test_inputs)
                zu.append(z.cpu().detach().numpy())
                test_data["pred"]=newDecoder(z)
                test_data = [post_transforms(i) for i in decollate_batch(test_data)]
                test_output = from_engine(["pred"])(test_data)


