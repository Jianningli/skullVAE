### Training β-VAE by Aggregating a Learned Gaussian Posterior with a Decoupled Decoder

## Code
```Python
zcr→co = zts + γDEVcr
zfa→co = zts + γDEVfa
```
![alt text](https://github.com/Jianningli/skullVAE/blob/main/figs/small_beta_output.png)



(1) train the initial VAE using beta=100 or beta=0.0001
```Python
python monaiSkullVAE.py --phase train
#python monaiSkullVAE.py --phase test
```

(2) train a decoder using the latent variables from the previous VAE under beta=100
```Python
python VAEDecoderRetrain.py --phase train
#python VAEDecoderRetrain.py --phase test
```

(3) make predictions with the aggregate VAE (encoder from beta=100 + decoupled decoder)  

```Python
python AggreegateVAE.py
```

## Dataset
Download the dataset [here](https://files.icg.tugraz.at/f/d06d433bd5f74f29ab8c/?dl=1).  
The dataset is extended from the [AutoImplant Challenge](https://autoimplant2021.grand-challenge.org/).
There are 100 healthy skulls, 100 skulls with facial and craial defects:

![alt text](https://github.com/Jianningli/skullVAE/blob/main/figs/dataset.png)

check out our other skull-reconstruction project with MONAI at [SkullRec](https://github.com/Jianningli/research-contributions/tree/master/SkullRec)

---
References:

Dataset
```
@inproceedings{li2020dataset,
  title={Dataset descriptor for the AutoImplant cranial implant design challenge},
  author={Li, Jianning and Egger, Jan},
  booktitle={Cranial Implant Design Challenge},
  pages={10--15},
  year={2020},
  organization={Springer}
}
```
Methods
```
@inproceedings{li2022training,
  title={Training β-VAE by Aggregating a Learned Gaussian Posterior with a Decoupled Decoder},
  author={Li, Jianning and others},
  booktitle={Medical Applications with Disentanglements (MAD)},
  year={2022},
  organization={Springer}
}
```
---
contact: feel free to contact jianningli.me[AT]gmail[dot]com, if you have questions about the codes and paper
