## Installation and Requirements

1.  **Clone this repository**:
    ```bash
    git clone https://github.com/Thomkat/MBC.git
    cd MBC
    ```

2.  **Create a new conda environment using the file provided**:
    ```bash
    conda env create -n mbc -f environment.yml
    ```
3.  **Activate the environment**:
    ```bash
    conda activate mbc
    ```

### Important Packages
```text
python==3.9.23
transformers==4.36.2
accelerate==0.30.0
hydra-core==1.3.2
bitsandbytes==0.47.0
wandb==0.21.4
scikit-learn
scipy
tqdm==4.67.1
torch==2.8.0
torchvision==0.23.0
```

-----

## Datasets

The project supports two types of dataset loading: from local CSV files and directly from the HuggingFace Hub.

### CSV Datasets (StreamingQA, ArchivalQA)

For the datasets `StreamingQA` and `ArchivalQA`, the code expects `.csv` files located in a directory specified in the corresponding dataset config file (e.g., `conf/dataset/streamingqa.yaml`).

The path is set using the `dataset.data_dir` parameter.

**Expected Structure:**

```
/path/to/your/datasets/
├── StreamingQA/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
│
└── ArchivalQA/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

### HuggingFace Datasets (SQuAD)

For the SQuAD dataset, the data is downloaded automatically from the HuggingFace Hub. The configuration file (`conf/dataset/squad.yaml`) specifies the splits and index ranges to be used for training, validation, and testing.