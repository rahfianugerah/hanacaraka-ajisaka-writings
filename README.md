![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/Pytorch-2.0+-red.svg)
[![MIT license](https://img.shields.io/badge/License-MIT-green)](LICENSE)
![Build](https://img.shields.io/badge/Build-Passing-green)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

### Hanacaraka Ajisaka Writings: Stacked Pretrained Models for Javanese Scripts Classification and OCR

<p align="justify">
This project implements an advanced optical character recognition (OCR) system for Javanese Aksara script, specifically targeting Aji Saka's historical writings including ꦲꦤꦕꦫꦏ, ꦣꦠꦱꦮꦭ, ꦥꦣꦗꦪꦚ, ꦩꦒꦧꦛꦔ. The system employs a stacked ensemble deep learning architecture combining multiple state-of-the-art convolutional neural networks with hyperparameter optimization for maximum accuracy.
</p>

### Supported Characters

<p align="justify">
The model recognizes and classifies the following Javanese Aksara character sets:
</p>

- ꦲꦤꦕꦫꦏ (HA NA CA RA KA)
- ꦣꦠꦱꦮꦭ (DA TA SA WA LA)
- ꦥꦣꦗꦪꦚ (PA DHA JA YA NYA)
- ꦩꦒꦧꦛꦔ (MA GA BHA THA NGA)

### Model Architecture

<p align="justify">
The system uses a Stacked Ensemble Model architecture that combines three pretrained deep learning backbones:
</p>

- **MobileNetV2**: Lightweight backbone producing 1280 features
- **ResNet50**: Deep residual network producing 2048 features
- **EfficientNet-B0**: Efficient scaling backbone producing 1280 features

<p align="justify">
These backbones are combined through feature concatenation (total 4608 features) and processed through a meta-learner neural network with configurable hidden units, batch normalization, ReLU activation, and dropout regularization.
</p>

### Technical Stack

### Core Dependencies

- **PyTorch**: Deep learning framework with CUDA 11.8 support
- **TorchVision**: Pretrained models and computer vision utilities
- **TorchAudio**: Audio processing capabilities
- **OpenCV**: Image processing and contour detection for character segmentation
- **Scikit-learn**: Train-test splitting, evaluation metrics, and classification reports
- **Optuna**: Tree-structured Parzen Estimator (TPE) for hyperparameter tuning
- **Timm**: PyTorch Image Models library
- **Matplotlib & Seaborn**: Data visualization and training curves
- **Pillow**: Image loading and preprocessing
- **NumPy & Pandas**: Numerical computing and data manipulation
- **PyTesseract**: Additional OCR capabilities
- **Kaggle**: Dataset download and management

### Experiments Devices

<p align="justify">
Platform: <a href="https://modal.com">modal.com</a>
</p>

- CPU: 4 cores
- RAM: 16GB
- GPU: Nvidia L4 24GB

### Installation

<p align="justify">
Install all required dependencies:
</p>

```bash
%uv pip install torch torchvision torchaudio
%uv pip install optuna timm opencv-python matplotlib scikit-learn pytesseract kaggle
```

### Dataset

<p align="justify">
The project uses the Hanacaraka dataset from Kaggle, which is automatically split into training (70%), validation (15%), and test (15%) sets with stratification to maintain class distribution.
</p>

### Download

<p align="justify">
Download the training dataset from Kaggle:
</p>

```bash
kaggle datasets download -d vzrenggamani/hanacaraka --force
```

### Extract

```python
import zipfile

zip_path = "/root/hanacaraka.zip"
extract_to = "/root/data"

with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(extract_to)
```

### Data Processing

<p align="justify">
The pipeline includes automated dataset cleaning to remove corrupted images, weighted random sampling to handle class imbalance, and comprehensive data augmentation for the training set including random rotation (15 degrees), random affine transformations, color jittering, and ImageNet normalization.
</p>

### Project Structure

```
hanacaraka-ajisaka-writings/
├── models.ipynb              # Main notebook with model implementations
├── README.md                 # Project documentation
└── test_ocr/                 # Test data and OCR samples
    ├── datasawala.jpeg
    ├── hanacaraka.jpeg
    ├── magabathanga.jpeg
    └── padhajayanya.jpeg
```

### Training Process

### Training Process

<p align="justify">
The model training employs advanced optimization techniques:
</p>

- **Mixed Precision Training**: Automatic mixed precision (AMP) with CUDA for faster training and reduced memory usage
- **Hyperparameter Optimization**: Optuna with Tree-structured Parzen Estimator optimizing learning rate (1e-5 to 1e-3), dropout rate (0.2 to 0.5), and hidden units (256, 512, 1024)
- **Optimizer**: AdamW optimizer with optimized learning rate from hyperparameter tuning
- **Loss Function**: CrossEntropyLoss for multi-class classification
- **Training Duration**: 30 epochs for final model training
- **Early Stopping**: Trial pruning mechanism for inefficient hyperparameter combinations

### Usage

<p align="justify">
The project is implemented as a Jupyter notebook (models.ipynb) containing the following sections:
</p>

### Library Setup

<p align="justify">
Installation and configuration of required dependencies including PyTorch with CUDA support, device detection, random seed initialization, and PIL configuration for truncated images.
</p>

### Data Loading and Preprocessing

<p align="justify">
Automated dataset scanning, corruption detection, train-validation-test splitting (70-15-15), weighted sampling for class balance, and augmentation pipeline setup with separate transforms for training and validation.
</p>

### Model Architecture Definition

<p align="justify">
Implementation of the StackedEnsembleModel combining MobileNetV2, ResNet50, and EfficientNet-B0 backbones with a configurable meta-learner head featuring batch normalization and dropout regularization.
</p>

### Hyperparameter Tuning

<p align="justify">
Optuna-based optimization with Tree Parzen Estimator sampling, automatic trial pruning, and 5-trial search for optimal learning rate, dropout rate, and hidden layer dimensions.
</p>

### Model Training

<p align="justify">
Full 30-epoch training with mixed precision, training and validation loss/accuracy tracking, real-time performance monitoring, and visualization of learning curves.
</p>

### Model Evaluation

<p align="justify">
Comprehensive evaluation on test set including classification reports with precision, recall, and F1-scores per class, confusion matrix analysis, and visual prediction demonstrations with confidence scores.
</p>

### OCR Pipeline

<p align="justify">
End-to-end OCR implementation featuring adaptive thresholding for binarization, contour-based character segmentation, letterbox preprocessing with aspect ratio preservation, batch processing capabilities, and word spacing detection based on character proximity.
</p>

### OCR Accuracy Testing

<p align="justify">
Automated accuracy calculation with space-invariant string matching, character-level similarity scoring using SequenceMatcher, exact match accuracy computation, and CSV report generation with detailed metrics.
</p>

### Model Persistence

<p align="justify">
Model saving functionality to preserve trained weights in PyTorch state dictionary format for deployment and inference.
</p>

### OCR Character Segmentation

<p align="justify">
The OCR system implements a sophisticated character segmentation pipeline:
</p>

- **Preprocessing**: Grayscale conversion and Gaussian blur for noise reduction
- **Binarization**: Adaptive thresholding with Gaussian weighting for varying lighting conditions
- **Segmentation**: External contour detection with noise filtering (minimum 10x10 pixels)
- **Sorting**: Left-to-right ordering of detected characters
- **Spacing Detection**: Configurable pixel distance threshold for word separation
- **Letterbox Padding**: Aspect ratio preservation during preprocessing to 224x224 resolution

### Model Evaluation Metrics

<p align="justify">
The models are evaluated using comprehensive metrics:
</p>

- **Accuracy Score**: Overall classification accuracy on test set
- **Classification Report**: Per-class precision, recall, F1-score, and support
- **Confusion Matrix**: Detailed error analysis across all character classes
- **Character Similarity**: Sequence matching ratio for OCR output evaluation
- **Exact Match Accuracy**: Percentage of perfectly recognized words (space-invariant)

### Running the Notebook

<p align="justify">
Follow these steps to execute the complete pipeline:
</p>

1. Ensure all dependencies are installed via the installation cell
2. Configure Kaggle API credentials for dataset download
3. Run data extraction and preprocessing cells
4. Execute model architecture definition
5. Run hyperparameter optimization (5 trials with Optuna)
6. Train final model with optimal parameters for 30 epochs
7. Evaluate model performance on test set
8. Upload test OCR images to test_ocr folder and extract
9. Run OCR pipeline for sentence recognition
10. Execute accuracy evaluation against ground truth
11. Save trained model to artifacts directory

<p align="justify">
Results and visualizations including training curves, confusion matrices, and OCR predictions will be generated during execution.
</p>

### Output

<p align="justify">
The notebook generates comprehensive outputs:
</p>

- **Training Metrics**: Epoch-by-epoch loss and accuracy for both training and validation sets
- **Visualization**: Training and validation accuracy/loss curves plotted using Matplotlib
- **Model Performance**: Classification report with precision, recall, and F1-score per character class
- **Confusion Matrix**: Visualized confusion matrix showing classification patterns
- **OCR Results**: Bounding box visualizations with predicted characters and confidence scores
- **Batch OCR Report**: Pandas DataFrame with filename, predicted text, and accuracy metrics
- **Accuracy Report**: CSV file with character similarity scores and exact match results
- **Model Artifacts**: Saved model weights in artifacts/aksara_jawa_stacked_model.pth

### Key Features

- **Stacked Ensemble Architecture**: Combines three pretrained CNN backbones for robust feature extraction
- **Automated Hyperparameter Tuning**: Uses Optuna TPE for optimal configuration search
- **Mixed Precision Training**: Leverages CUDA AMP for efficient GPU utilization
- **Class Imbalance Handling**: Weighted random sampling ensures balanced training
- **Comprehensive Data Augmentation**: Multiple augmentation techniques prevent overfitting
- **End-to-End OCR Pipeline**: From raw image to recognized text with segmentation
- **Batch Processing**: Supports multiple image OCR with automated reporting
- **Adaptive Preprocessing**: Handles varying lighting and image quality conditions
- **Model Persistence**: Easy save and load functionality for deployment

### Notes

- GPU acceleration requires CUDA 11.8 compatible hardware
- Mixed precision training requires compatible GPU (Volta architecture or newer)
- Batch size (32) and image size (224x224) can be adjusted based on available GPU memory
- OCR test images must contain Javanese Aksara words or sentences for proper segmentation
- Space threshold for word separation (20 pixels) can be tuned based on character spacing
- Dataset cleaning automatically removes corrupted or unreadable images
- Model training takes approximately 30 epochs for convergence
- Hyperparameter tuning with 5 trials takes additional time before final training
- PyTesseract integration available for additional OCR capabilities
- Dataset is sourced from Kaggle community contributions

### License

<p align="justify">
This project is licensed under the MIT License.
</p>

### Contributor
<p align="justify">
 Thank you for contributing in this project, maybe next future work can be improved with another method, ready-use pipeline for production and more robust algorithm. 
 Below is the contributor of this project, big thanks, love and full support from <code>The Engineers</code>:
</p>

<div align="center">
 
| Contributor | GitHub |
| --- | --- |
| Naufal Rahfi Anugerah | [@rahfianugerah](https://www.github.com/rahfianugerah) |
| Achmad Ardani Prasha | [@achmadardhanip](https://www.github.com/achmadardanip) |
| Clavino Ourizqi Rachmadi | [@clavinorach](https://www.github.com/clavinorach) |

</div>

### Citation

```bash
@software{
   hanacaraka_stacked_pretrained_models_2025,
   title={Stacked Pretrained Models for Javanese Scripts Classification and OCR},
   author={Naufal Rahfi Anugerah, Achmad Ardani Prasha, Clavino Ourizqi Rachmadi},
   year={2025},
   url={https://github.com/rahfianugerah/hanacaraka-ajisaka-writings}
}
```