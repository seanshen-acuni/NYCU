# Semi-Supervised Animal Image Classification (CNN & RNN)

## Description
Two complementary PyTorch implementations for 10-class animal image classification under a semi-supervised setting:
- **CNN (ResNet18)** with confidence-based pseudo-labeling on unlabeled data.
- **RNN (LSTM) baseline** that treats images as sequences and performs pseudo-labeling + retraining.

Both pipelines resize images to **128×128** and run on GPU when available.

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

## Methods
- **CNN (ResNet18)**  
  - Train on labeled → generate pseudo-labels for **unlabeled** with a **probability threshold 0.95** → merge and retrain → predict (includes unlabeled + private test).
- **RNN (LSTM)**  
  - Flatten image tensor to a sequence → LSTM → FC → train on labeled → **generate pseudo-labels for unlabeled (no threshold)** → merge and retrain → predict (unlabeled + private test; also provides a Kaggle-style CSV). 

## Files
- `image_classification_cnn.py` — CNN (ResNet18) semi-supervised pipeline; writes `sample_submission.csv` with `ID,label`. 
- `image_classification_rnn.py` — RNN (LSTM) semi-supervised pipeline; saves `sample_submission.csv` and `sample_submission_kaggle.csv` (and a local `model.pth`, not included in repo). 
- `cnn_result.csv`, `rnn_result.csv`, `rnn_kaggle.csv` — Example prediction files produced on `private_test_data` (and, for Kaggle format, as scripted in code).

## Requirements
- Python 3.9+
- PyTorch, torchvision, tqdm, pandas, Pillow, numpy

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
- CNN uses **ResNet18** backbones; the script collects high-confidence unlabeled samples with probability > 0.95 and retrains.
- RNN uses a simple **LSTM** with `input_size=128`, `hidden_size=128`, `output_size=10`. It generates pseudo-labels for all unlabeled samples (no threshold).
- The RNN script writes two CSVs: a Kaggle-style file and a general submission file, both with `ID,label` columns sorted by numeric ID.

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
