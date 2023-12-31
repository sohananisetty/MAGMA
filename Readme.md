# MAGMA - Pytorch Implementation
## Music Aligned Generative Motion Autodecoder

Mapping music to dance is a challenging problem that requires spatial and temporal coherence along with a continual synchronization with the music’s progression. Taking inspiration from large language models, we introduce a 2-step approach for generating dance using a Vector Quantized-Variational Autoencoder (VQ-VAE) to distill motion into primitives and train a Transformer decoder to learn the correct sequencing of these primitives. We also evaluate the importance of music
representations by comparing naive music feature extraction using Librosa to deep
audio representations generated by state-of-the-art audio compression algorithms.
Additionally, we train variations of the motion generator using relative and absolute
positional encodings to determine the effect on generated motion quality when
generating arbitrarily long sequence lengths. Our proposed approach achieve
state-of-the-art results in music-to-motion generation benchmarks and enables
the real-time generation of considerably longer motion sequences, the ability to
chain multiple motion sequences seamlessly, and easy customization of motion
sequences to meet style requirements.

## QuickStart

### Creating environment

```.bash
conda env create -f environment.yml
```

### Download models

* Download the VQVAE model checkpoint from [Google Drive](https://drive.google.com/file/d/1GXpw0rUMiLXoKyzt3yl0dZJijJjnfeLA/view?usp=share_link) and place it in `./checkpoints/vqvae/mix/`. 

* Download the MotionSeq Decoder model checkpoint from [Google Drive](https://drive.google.com/file/d/1gsRS7GWIHJZXX8ZkC8pGLUCamQNU4Jwq/view?usp=share_link) and place it in `./checkpoints/motionseq/encodec/`. 


Or run `bash download_model.sh` in `./prepare/`.

### Generate motion

* Run `MAGMA.ipynb`, to load music, encode it and generate novel motions using the decoder.


### Sample motions

The results folder contains example motions generated from both music in the AIST++ dataset and in-th-wild music


## Acknowledgements

We would like to thank [lucidrains](https://github.com/lucidrains/x-transformers) for the ALiBi and parts of the transformer implementation, [Encodec](https://github.com/facebookresearch/encodec) for music compression code, and [T2M-GPT](https://github.com/Mael-zys/T2M-GPT) for code to render SMPL meshes.
