# diffusion-noise-cls-pytorch

This is an unofficial implementation of the diffusion-style noise frontend in ["Intriguing properties of generative classifiers" by Priyank Jaini, Kevin Clark, Robert Geirhos](https://arxiv.org/abs/2309.16779) to improve the shape-bias of vision models.

## Training

Reproduce the training of ResNet-50 (assumes you have 4 GPUs on 1 node)
```bash
torchrun --nproc-per-node=4 train.py --amp --no-normalization --diffusion-noise
```

If you have a different number of GPUs, adjust the batch size (via `--batch-size B`) such that `B * GPUs = 256`.

Please note that the resulting model is not normalized by channel means/stds, i.e., you must ensure that input data is in the `[0, 1]` range.

## Results

| Model                           	| ImageNet-val 	| Shape-Bias 	| Mean OOD 	|
|---------------------------------	|--------------	|------------	|----------	|
| RN-50 (90 Epoch) - eval with noise  	| 51.64 %        	| 0.73       	| 46.63 %   	|
| RN-50 (90 Epoch) - eval w/o noise 	| 67.22 %      	| 0.51       	| 56.06 %   	|

Please note that the evaluation with noise will give you slightly different results based due to the non-deterministic nature of noise.

#### Differences to original publication/codebase
The original codebase is written in JAX, trains for 300 epochs and does not use AMP - as such the results do not fully align.


## Credits
Thanks to Robert Geirhos for the help in reproducing the results! The training script is based on [the example in PyTorch](https://github.com/pytorch/vision/tree/main/references/classification).
