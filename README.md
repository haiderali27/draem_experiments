# draem_experiments

This repository reuses [DRAEM-PytorchLightning](https://github.com/juanb09111/DRAEM-PytorchLightning) which is [DRAEM's](https://github.com/VitjanZ/DRAEM) PytochLightning implementation. This repository adds [modifications](#modifications) to the original implementation. This implementation is the part of Master's thesis [AUTOMATED CAR DAMAGE DETECTION: A NOVEL APPROACH](https://trepo.tuni.fi/handle/10024/159865)

## Usage

**1. Clone Repository**.

`git clone https://github.com/haiderali27/anomaly_dataset_pipeline.git`

or

`git clone git@github.com:haiderali27/anomaly_dataset_pipeline.git`

**2. Install Requirements**.

It is recommended to use an isolated virtual environment with python version 3.8.

`pyenv virtualenv 3.8 draem`

`pyenv activate draem`

`pip install -r requirements.txt`

**3. Train**.

`python draem.py train --dataset-path <anomaly-dataset-path> --object-name <anomaly-object-name>`

**4. Test**.

Without existing weights `python draem.py test --dataset-path <anomaly-dataset-path> --object-name <anomaly-object-name>`

or

With existing weights `python draem.py test --dataset-path <anomaly-dataset-path> --object-name <anomaly-object-name> --load-checkpoints True --checkpoint-path <weights-path>`

There should be only two weights files in the path and the files should come from training the model. Don't change weight file names. Weight names are parsed before models are loaded to choose the right model.

## Modifications

The modification diagrams can be seen in section 3.4.

### Augmentation

Augmentations on black pixels with the zero value can be ignored in training. [Notebook](augmentation-demo.ipynb) shows the comparison between original and this implementation.

### Reconstruction Network

A new layer is added to the output of original draem reconsturction network before it reconstructs the image. The new layers can be one of these following.

**VAE-Encoder** - Variational Auto-encoder - Adds a variationl-auto-encoder layer

**Attention-Encoder** - Attention encoder - Adds an attention layer

**VAE-Attention-Encoder** - Variational Attention encoder - Adds an Variational-auto-encoder and Attention Layer in the chronological order

### Loss function

Draem use Structural Similarity Index Measure (SSIM) loss function for reconstruction network. The new reconstruction loss functions can be one of following.

**LPIPS** - Learned Perceptual Image Patch Similarity

**MPIPS** - Multi Scale Structural Similarty

**Usage Example:**

In this example VAE-Attention-Encoder reconstruction network is used with LPIPS reconstruction loss. Augmentations on black pixels are ignored. The anomaly-object in this example is `cars`.

`python draem.py train --dataset-path <anomaly-dataset-path> --object-name cars --reconstruction-network-name VAE-Attention-Encoder --reconstruction-loss LPIPS --ignore-black-region True`

`--help` flag provides details of all arguments that can be used to set learning-rate, training-epochs, batchset etc.

`python draem.py train --help`

`python draem.py test --help`

## Results

Results can be seen in section 4.5.
