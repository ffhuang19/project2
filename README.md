# Road Segmentation using U-Net

This project aims to identify and segment roads from aerial satellite images. It uses a U-Net architecture with a pre-trained encoder from the `segmentation-models-pytorch` library to achieve high accuracy. The final output is a `submission.csv` file that classifies 16x16 patches of the test images as "road" or "background".

## Features

-   **Modern Architecture**: Implements a U-Net with a powerful, pre-trained ResNet34 encoder for effective feature extraction.
-   **Automated Pipeline**: A single script (`run.py`) handles the entire workflow: training, validation, saving the best model, and generating the final submission file.
-   **Robust Training**: Includes strong data augmentation and a combined Dice + BCE loss function to improve model generalization and stability.
-   **Optimized for Performance**: Automatically uses a GPU if available (`cuda` device) and is configured for efficient data loading.

## Prerequisites

-   Python 3.8+
-   An NVIDIA GPU with CUDA support is **highly recommended** for training in a reasonable amount of time. The script will fall back to CPU, but it will be extremely slow.
-   The project dataset.

## Setup and Installation

Follow these steps to set up the environment and run the project.

**1. Project Structure**

Place the unzipped dataset folders (`training` and `test_set_images`) in the same directory as the `run.py` script. The expected file structure is:

```
.
├── run.py
├── training/
│   ├── images/
│   │   ├── satImage_001.png
│   │   └── ...
│   └── groundtruth/
│       ├── satImage_001.png
│       └── ...
├── test_set_images/
│   ├── test_1/
│   │   └── test_1.png
│   ├── test_2/
│   │   └── test_2.png
│   └── ...
└── README.md
```

**2. Create a Virtual Environment (Recommended)**

It's best practice to create a separate Python environment for the project to avoid conflicts with other libraries.

```bash
# Create a new virtual environment
python3 -m venv .venv

# Activate the environment
# On macOS and Linux:
source .venv/bin/activate
# On Windows:
# .\.venv\Scripts\activate
```

**3. Install Dependencies**

This project requires several Python libraries. You can install them all using the following `pip` command:

```bash
pip install torch torchvision torchaudio
pip install segmentation-models-pytorch albumentations opencv-python tqdm
```

**4. Pre-download Model Weights (Optional but Recommended)**

The first time you run the script, it will try to download pre-trained weights for the ResNet34 model. This can sometimes fail due to network or SSL issues. To prevent this, you can download the weights manually beforehand.

-   **Download the file:** [resnet34-333f7ec4.pth](https://download.pytorch.org/models/resnet34-333f7ec4.pth)
-   **Create the cache directory:**
    ```bash
    # On macOS or Linux
    mkdir -p ~/.cache/torch/hub/checkpoints/
    ```
-   **Move the downloaded file** (`resnet34-333f7ec4.pth`) into the `~/.cache/torch/hub/checkpoints/` directory.

## Usage

The `run.py` script is designed to be an all-in-one solution.

**To start the process, simply run the script from your terminal:**

```bash
python run.py
```

The script will execute the following steps automatically:
1.  **Start Training**: It will load the training data, set up the U-Net model, and begin the training and validation process for the number of epochs defined in the script.
2.  **Save Best Model**: During training, it will continuously save the version of the model that achieves the highest F1-score on the validation set to a file named `best_model.pth`.
3.  **Generate Submission**: After training is complete, it will load `best_model.pth`, run predictions on all images in the `test_set_images/` directory, and generate the final `submission.csv` file.
