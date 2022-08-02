# skull_vae





(1)training the initial VAE using beta=100 or beta=0.0001
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



check out our other skull project with MONAI at [SkullRec](https://github.com/Jianningli/research-contributions/tree/master/SkullRec)
