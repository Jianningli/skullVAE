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
    SaveImaged,
    Zoomd,
)

from monai.handlers.utils import from_engine
#from monai.networks.nets import VarAutoEncoder
from vae import VarAutoEncoder
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss,DiceCELoss
from monai.inferers import Inferer, SimpleInferer
from monai.data import DataLoader, Dataset, decollate_batch
from monai.optimizers import Novograd
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import torch
import numpy as np
import argparse


'''########## Dataset diretory
'''


directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)


# 'train_images' and 'train_labels' are the same

train_images = sorted(
    glob.glob(os.path.join( "./vae_dataset/train/input/", "*.nii.gz")))


train_labels = sorted(
    glob.glob(os.path.join( "./vae_dataset/train/gt/", "*.nii.gz")))


data_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_images, train_labels)
]




train_files = data_dicts


# Since VAE training is unsupervised, i.e., the input and output are 
# the same. make sure to combine both the training and test set, 
# if applicable, while training the VAE

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
        #SaveImaged(keys=["image"],output_dir='./train_transformed_data')
    ]
)


train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Resized(keys=["image", "label"],spatial_size=(256, 256, 128),mode='nearest'),
        EnsureTyped(keys=["image", "label"]),
        #SaveImaged(keys=["image"],output_dir='./train_transformed_data')
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
    SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir="./vae_output", output_postfix="completed", resample=False),
])





'''########## Load datasets and apply transforms
'''

train_ds = Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4)
#val_ds = Dataset(data=val_files, transform=val_transforms)
#val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)
test_org_ds = Dataset(data=test_data, transform=test_org_transforms)
test_org_loader = DataLoader(test_org_ds, batch_size=1, num_workers=4)



'''########## Network and training specifications
'''

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



loss_function = DiceLoss(to_onehot_y=True, softmax=True,reduction='sum') 
optimizer = torch.optim.Adam(model.parameters(), 1e-4)
dice_metric = DiceMetric(include_background=True, reduction="mean")


max_epochs = 200


# beta=0.0001
beta = 100


val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
dice_loss_values=[]
kld_loss_values=[]
metric_values = []
post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])

 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase")
    args = parser.parse_args()
    if args.phase=='train':
         
        print('**********************start traininig*************************')

        #uncomment to start training from the previous weights

        #weights_dir='./vae_weights/'
        #model.load_state_dict(torch.load(os.path.join(weights_dir, "final_epoch_model.pth"),map_location='cuda:0'))
         
        for epoch in range(max_epochs):
 
            print(" -" * 10)
            print(f"epoch {epoch + 1}/{max_epochs}")
            model.train()
            epoch_loss = 0
            dice_loss=0
            kld_loss=0
            step = 0
            for batch_data in train_loader:
                step += 1
                inputs, labels = (  
                    batch_data["image"].to(device),   
                    batch_data["label"].to(device),
                )
                optimizer.zero_grad()
                recon_batch, mu, log_var, _ = model(inputs)

                bceloss = 80*loss_function(recon_batch, labels)
                print('dice loss:',bceloss.item()) 
                kldloss = -0.5 * beta * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                print('KLD loss:',kldloss.item())

                loss=bceloss+kldloss

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                dice_loss += bceloss.item()
                kld_loss += kldloss.item()

                print(
                    f"{step}/{len(train_ds) // train_loader.batch_size}, "
                    f"overall train_loss: {loss.item():.4f}")
            epoch_loss /= step
            dice_loss /= step
            kld_loss /= step

            epoch_loss_values.append(epoch_loss)
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
            dice_loss_values.append(dice_loss)
            print(f"epoch {epoch + 1} dice loss: {dice_loss:.4f}")
            kld_loss_values.append(kld_loss)
            print(f"epoch {epoch + 1} kld loss: {kld_loss:.4f}")

        torch.save(model.state_dict(), os.path.join(root_dir, "final_epoch_model.pth"))

        epoch_loss_values=np.array(epoch_loss_values)
        np.save('overall_beta100_loss.npy',epoch_loss_values)

        dice_loss_values=np.array(dice_loss_values)
        np.save('dice_beta100_loss.npy',dice_loss_values)

        kld_loss_values=np.array(kld_loss_values)
        np.save('kld_beta100_loss.npy',kld_loss_values)


    elif args.phase=='test':
        print('**************generating predictions on the test set***************')
        latent_var=[]
        sigma_var=[]
        zu=[]
        weights_dir='./vae_weights/vae_weights/'
        model.load_state_dict(torch.load(os.path.join(weights_dir, "final_epoch_model.pth"),map_location='cuda:0'))
        model.eval()
        with torch.no_grad():
            for test_data in test_org_loader:
                test_inputs = test_data["image"].to(device)
                inferer=SimpleInferer()
                test_data["pred"],u,sigma,z= inferer(test_inputs, model)
                latent_var.append(u.cpu().detach().numpy())
                sigma_var.append(sigma.cpu().detach().numpy())
                zu.append(z.cpu().detach().numpy())

                print(u.cpu().detach().numpy().shape)
                test_data = [post_transforms(i) for i in decollate_batch(test_data)]
                test_output = from_engine(["pred"])(test_data)
                print(test_output[0].detach().cpu().shape)



            #  save the latent output 

            '''
            zu=np.array(zu)
            print(zu.shape)
            np.save('zu_beta100.npy',zu)


            latent_var=np.array(latent_var)
            print(latent_var.shape)
            np.save('var_beta100.npy',latent_var)

            sigma_var=np.array(sigma_var)
            print(sigma_var.shape)
            np.save('sigma_beta100.npy',sigma_var)
            '''

    
