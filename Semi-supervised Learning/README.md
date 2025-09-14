# Semi-Supervised Animal Image Classification (CNN & RNN)

## Description
Two complementary PyTorch implementations for 10-class animal image classification under a semi-supervised setting:
- **CNN (ResNet18 backbone)** with confidence-based pseudo-labeling on unlabeled data.
- **RNN (LSTM)** baseline that treats images as sequences and applies aggressive pseudo-labeling.

Both pipelines resize images to **128×128** and run on GPU when available.

## Methods
- **CNN (ResNet18)**  
  - Train the backbone on labeled_data for 1 epoch.
  - Run inference on unlabeled_data only to generate pseudo-labels, keeping samples with probability > 0.95.
  - Retrain on the union of labeled_data + high-confidence pseudo-labeled unlabeled_data.
  - Perform final inference on the combined dataset (labeled_data + unlabeled_data + private_test_data) and save predictions to CSV.

- **RNN (LSTM)**  
  - Reshape each 128×128 image into a sequence of 128 steps × 128 features.
  - Train on labeled_data for 10 epochs.
  - Run inference on unlabeled_data only and assign pseudo-labels to all samples (no threshold).
  - Retrain on the union of labeled_data + all pseudo-labeled unlabeled_data.
  - Final inference on all three splits (labeled_data, unlabeled_data, private_test_data); outputs both a general CSV and a Kaggle-style CSV.


## Dataset layout
The code expects this structure:
```
.
└── datasets/
    ├── labeled_data/
    │   ├──cat
    │   ├──bird
    │   ├──dog
    │   ├──deer
    │   ├──...
    │   └──wolf
    ├── unlabeled_data/
    └── private_test_data/
```
Class mapping (ID order) is defined in code: dog=0, wolf=1, …, duck=9.

## Files
- `image_classification_cnn.py` — CNN (ResNet18) semi-supervised pipeline; writes `sample_submission.csv` with `ID,label`. 
- `image_classification_rnn.py` — RNN (LSTM) semi-supervised pipeline; saves `sample_submission.csv` and `sample_submission_kaggle.csv` (and a local `model.pth`, not included in repo). 
- `cnn_result.csv`, `rnn_result.csv`, `rnn_kaggle.csv` — Example prediction files produced on `private_test_data` (and, for Kaggle format, as scripted in code).

## Requirements
- Python 3.11+
- PyTorch
- torchvision
- tqdm
- pandas
- Pillow
- numpy

Install (example):
```
pip install torch torchvision tqdm pandas pillow numpy
```

## Usage
1. Prepare dataset folders (see “Dataset layout” above).
2. Run CNN pipeline: 
```
python image_classification_cnn.py
```
3. Run RNN pipeline:
```
python image_classification_rnn.py
```
Both scripts read images from datasets/unlabeled_data and datasets/private_test_data and write CSV predictions (IDs are file names).

## Results & Outputs
- Example CSVs are included in the repo:
  - `cnn_result.csv` — CNN predictions on `private_test_data`.
  - `rnn_result.csv` — RNN predictions on `private_test_data`.
  - `rnn_kaggle.csv` — Kaggle-style CSV as produced by the RNN script.

## Notes
- The CNN baseline is conservative: it filters pseudo-labels strictly and only trained for one epoch.
- The RNN, while unconventional for images, trained longer (10 epochs) and leveraged all unlabeled data, giving it an edge in practice.
- As a result, the RNN outperformed the CNN in final accuracy, even though CNN is typically the stronger architecture for vision tasks.

## Support
If you encounter issues or have questions:
- Open a GitHub Issue
- Contact: 40924seanshen@gmail.com

## Contributing
Contributions are welcome!

1. Fork this repository
2. Create a new branch: git checkout -b feature/my-feature
3. Commit your changes
4. Open a Pull Request

## Author
Sean Shen (沈昱翔)

## License
This project is licensed under the MIT License.

## Project Status
Completed – Implemented, trained, and evaluated CNN & RNN semi-supervised pipelines; predictions generated for the `private_test_data` split.
