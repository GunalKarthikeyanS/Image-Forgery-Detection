# Image Forgery Detection and Localization

Pixel-level detection of tampered regions in images with U-Net and Swin-UNet models, trained on the RECODAI dataset. Each model outputs a mask marking where an image was manipulated, and the same output decides whether the whole image is authentic or forged.

## What's in this repository

- **`src/`**: a corrected data pipeline, U-Net models, losses, metrics and training loop
- **`notebooks/train_and_evaluate.ipynb`**: trains and compares six setups on Colab and reports test results
- **`legacy/`**: the original notebooks and code, kept unchanged for reference

## Experiments

| Experiment | Model | Input | Loss |
|---|---|---|---|
| `unet_bce` | U-Net | RGB | Binary cross-entropy |
| `unet_combined` | U-Net | RGB | BCE + Dice + IoU |
| `unet_deepsup_combined` | U-Net with five output heads (deep supervision) | RGB | BCE + Dice + IoU on every head |
| `unet_combined_ela` | U-Net | RGB + ELA | BCE + Dice + IoU |
| `swin_unet_combined` | Swin-UNet | RGB | BCE + Dice + IoU |
| `swin_unet_combined_ela` | Swin-UNet | RGB + ELA | BCE + Dice + IoU |

- **U-Net** (about 26M parameters): residual convolution blocks with instance normalization and PReLU, bilinear down- and up-sampling, additive skip connections.
- **Swin-UNet**: an ImageNet-pretrained Swin-Tiny transformer encoder (via `timm`) with a convolutional U-Net decoder and concatenated skip connections.
- **ELA (Error Level Analysis)**: each image is re-saved as JPEG at quality 90 and the amplified difference is added as three extra input channels. Edited regions often re-compress differently from the rest of the image.

All models train at 448×448 with AdamW (learning rate 5e-4, weight decay 1e-4), cosine decay, mixed precision, and JPEG-compression, colour-jitter and flip augmentation. The checkpoint with the best validation Dice is kept.

## Evaluation

The test set holds out 20% of images, **both authentic and forged**. Images sharing a file name stay in the same split, so nothing leaks between training and testing.

- **Localization** (forged images): Dice and IoU between the predicted and true forged regions
- **Detection** (all images): accuracy, F1 and ROC-AUC for authentic vs forged, scoring each image by its most confident predicted pixels

## Results

<!-- Fill in after running notebooks/train_and_evaluate.ipynb -->

| Experiment | Dice | IoU | Accuracy | F1 | ROC-AUC |
|---|---|---|---|---|---|
| `unet_bce` | | | | | |
| `unet_combined` | | | | | |
| `unet_deepsup_combined` | | | | | |
| `unet_combined_ela` | | | | | |
| `swin_unet_combined` | | | | | |
| `swin_unet_combined_ela` | | | | | |

## Fixes compared with the original code

The original notebooks (in `legacy/`) reported only Dice/IoU, and their best logged run reached Dice 0.40 / IoU 0.29. That evaluation had several problems, corrected here:

- **Masks matched by order, not name.** Forged images and masks were paired by `os.listdir` order, which is arbitrary, so images could be trained against the wrong mask. They are now matched by file name.
- **Forged-only test set.** The image list was ordered authentic-then-forged and split by position, so the test set contained no authentic images. Splits are now random and include both.
- **Augmented, shuffled test data.** Test scores changed between runs; evaluation is now deterministic.
- **Deep supervision trained one head.** The five-head model only trained its lowest-resolution head; all heads are now trained, and predictions use the finest one.
- **Blurred masks.** Masks were resized with bicubic interpolation; they now use nearest-neighbour.
- **No detection metrics.** Whole-image accuracy, F1 and ROC-AUC are now reported.

The original notebooks also tried CAT-Net and MicroSegNet. Both are medical-imaging models (prostate MRI and micro-ultrasound; this CAT-Net is unrelated to the forgery-detection CAT-Net of the same name) and were trained from scratch, so they are replaced here by a pretrained Swin-UNet.

## How to run

1. Put the dataset on Google Drive in this layout:
   ```
   image_forgery/data/recodai/
   ├── train_images/
   │   ├── authentic/     (images)
   │   └── forged/        (images)
   └── train_masks/       (one .npy mask per forged image, same file name)
   ```
2. Open `notebooks/train_and_evaluate.ipynb` in Google Colab with a GPU runtime and run all cells. Results and checkpoints are saved to `image_forgery/runs/` on Drive.

## Limitations

- Results come from a single random split and one training run per setup.
- Images are padded to a square of 448×448, which shrinks large images and can hide small forged regions.
- ELA is designed for JPEG images; on images that were never JPEG-compressed it carries less signal.
- Detection is derived from the segmentation output with a fixed threshold of 0.5.

## Credits

Team project by **Gunal Karthikeyan Saravanan** and **Hyeonwook Kim**.

- Original model code and training notebooks (in `legacy/`): Hyeonwook Kim
- Corrected data pipeline, evaluation, ELA, Swin-UNet and experiments (`src/`, `notebooks/`): Gunal Karthikeyan Saravanan
