# CBD stone classification on MRCP

This project uses ResNet-50 to classify MRCP images into two groups:

- `0_no_stone`: no CBD stone.
- `1_CBD_stone`: CBD stone present.

MRCP stands for magnetic resonance cholangiopancreatography. CBD means common bile duct.

## Notebooks

| Notebook | Purpose |
| --- | --- |
| [MRCP_Resnet50.ipynb](MRCP_Resnet50.ipynb) | Train the classifier and save the best model by validation accuracy. Evaluate the internal and external test sets at a cutoff of 0.5. Also plot exploratory F1 curves. |
| [Best_Thr_cropped.ipynb](Best_Thr_cropped.ipynb) | Load the cropped-image model. Find the threshold with the highest validation F1, then apply that same threshold to both test sets. |
| [Best_Thr_ Non_Cropped.ipynb](Best_Thr_%20Non_Cropped.ipynb) | The same threshold analysis using non-cropped images and their trained model. |

The two threshold notebooks differ only in their paths and descriptions. Neither trains a model or crops images. Cropped images must already exist before running the cropped analysis.

## Data and paths

The notebooks expect this folder layout in Google Drive:

```text
MyDrive/
├── mrcp_data/
│   ├── non_cropped/
│   │   ├── internal/
│   │   │   ├── 0_no_stone/
│   │   │   └── 1_CBD_stone/
│   │   └── external/
│   │       ├── 0_no_stone/
│   │       └── 1_CBD_stone/
│   └── cropped/
│       ├── internal/
│       │   ├── 0_no_stone/
│       │   └── 1_CBD_stone/
│       └── external/
│           ├── 0_no_stone/
│           └── 1_CBD_stone/
└── mrcp_outputs/
    ├── non_cropped/
    └── cropped/
```

Put the image files inside the class folders. The code loads ordinary image files through `torchvision.datasets.ImageFolder`; it does not contain a DICOM preprocessing pipeline.

Change `BASE_DIR` and `OUTPUT_DIR` in the paths cell if your folders are elsewhere. Internal and external folders must use the same class names and label order. The images and trained `mrcp_resnet50.pth` files are not included in this repository.

## How to run

1. Open `MRCP_Resnet50.ipynb` in Google Colab. Use a GPU runtime for training if available.
2. Run the installation cell, mount Google Drive, and check the paths.
3. Run the remaining cells in order. The notebook saves `mrcp_resnet50.pth` in `OUTPUT_DIR`.
4. Open the matching threshold notebook and point it to the same dataset and output folder.
5. Run its cells in order to select the validation threshold and evaluate both test sets.

For a cropped model, change both training paths from `non_cropped` to `cropped`, train separately, then use `Best_Thr_cropped.ipynb`. A separate cropped training notebook was not supplied. Keep the two models in separate output folders.

The installation cell includes PyTorch, torchvision, scikit-learn, NumPy, matplotlib and Pillow. Google Drive mounting uses Colab's built-in integration. A local Jupyter run requires replacing the Drive mount and paths.

## Training setup

| Setting | Value |
| --- | --- |
| Model | ResNet-50 with ImageNet V2 weights and one output logit |
| Internal split | Approximately 70% training, 20% validation, 10% test; stratified by class |
| External data | Evaluated separately without splitting |
| Random seed | 42 |
| Image size | 224 × 224 |
| Training augmentation | Random horizontal flip and rotation up to 10 degrees |
| Normalization | ImageNet mean and standard deviation |
| Loss | Binary cross-entropy with logits |
| Optimizer | Adam, learning rate `1e-4`; all model parameters are trained |
| Scheduler | StepLR, step size 7, gamma 0.1 |
| Epochs | 20 |
| Batch size | 32 |
| Checkpoint selection | Highest validation accuracy at threshold 0.5 |

The project author reports one image per patient and separation of patients between sets. The notebooks themselves split image indices; they do not read patient IDs or an existing split manifest. The threshold notebooks reconstruct that split using the same seed and file ordering. Keep the image collection and folder names unchanged between training and threshold analysis. Patient separation cannot be independently checked from the supplied files.

## Outputs

The training notebook saves:

- `mrcp_resnet50.pth`: the best model's state dictionary.
- `mrcp_metrics.json`: internal and external test metrics.
- Confusion matrices, ROC curves, precision–recall curves and F1-versus-threshold plots.

Each threshold notebook saves `mrcp_metrics_valthr.json`, validation plots and test plots using the validation-selected threshold. Both notebooks use the same output filenames in different folders. Re-running a notebook in the same output folder replaces those outputs.

The training notebook's `accuracy` uses threshold 0.5, while its `best_f1` and `best_threshold` come from an exploratory search on the evaluated set. Those fields describe different cutoffs. Use the threshold notebooks for results obtained with a cutoff chosen on validation data before test evaluation.

In the threshold JSON files, `internal_val_threshold` means **internal test performance using the validation threshold**. It does not mean validation-set performance. `pr_auc` is the trapezoidal area under the precision–recall curve, not average precision.

## Supplied results

These numbers are copied from the supplied `Cropped.json` and `Non-Cropped.json`, without recalculation or retraining. The original JSON files are not included in this code repository.

| Images | Validation threshold | Internal test accuracy | Internal test F1 | External test accuracy | External test F1 | External ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cropped | 0.569861 | 0.998791 | 0.998870 | 0.960000 | 0.961538 | 0.999900 |
| Non-cropped | 0.281433 | 0.968487 | 0.969072 | 0.850000 | 0.864865 | 0.969300 |

The original notebook logs report 8,268 internal images for the cropped analysis and 4,751 for non-cropped, with 200 external images in each. These are different-sized internal datasets, so the table alone does not isolate the effect of cropping. The code does not establish how the two image collections correspond.

Exact results depend on the data, checkpoint, library versions and hardware. The supplied notebooks did not record a complete versioned environment, and setting the seed alone does not guarantee identical GPU runs.
