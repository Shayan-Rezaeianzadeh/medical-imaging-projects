# Medical imaging projects

Two projects: classifying common bile duct stones on MRCP images, and predicting worsening on a second brain CT that leads to surgery.

## Files

| File | What it does |
| --- | --- |
| [MRCP_Resnet50.ipynb](mrcp/MRCP_Resnet50.ipynb) | Trains a ResNet-50 classifier for CBD stones. The default paths use non-cropped images. |
| [Best_Thr_cropped.ipynb](mrcp/Best_Thr_cropped.ipynb) | Loads the cropped-image model, selects a threshold on validation data, and evaluates the internal and external test sets. |
| [Best_Thr_ Non_Cropped.ipynb](mrcp/Best_Thr_%20Non_Cropped.ipynb) | Runs the same threshold analysis for non-cropped images. |
| [Tree Models.py](brain_ct/Tree%20Models.py) | Compares XGBoost, Random Forest, Decision Tree and Gradient Boosting for the brain CT outcome. Exports metrics, plots, feature importance and SHAP results. |
| [Non Tree Models.py](brain_ct/Non%20Tree%20Models.py) | Compares MLP, K-Nearest Neighbors, Gaussian Naive Bayes and SVM for the same outcome. Exports metrics and ROC plots. |

The MRCP files are Google Colab notebooks. The brain CT files are Python scripts with `# %%` sections for running in an editor such as Spyder or VS Code.

## Run a project

- **MRCP:** follow the [MRCP README](mrcp/README.md). Train a model first, then run the matching threshold notebook.
- **Brain CT:** follow the [brain CT README](brain_ct/README.md). Supply `data1.csv`, install the requirements, and run either script.

The image datasets, trained MRCP weights and brain CT input CSV are not included. This repository contains the code and documentation; saved result files are not included.

## Saved results

The MRCP README summarizes the supplied cropped and non-cropped metrics. The brain CT README describes the saved experiment versions. Only the latest SMOTE-then-ADASYN scripts were supplied; the source for the earlier experiment versions is not available here.

These are research analyses. See each project's README for the split, threshold selection and interpretation of the saved results.
