# Spam SMS Detection with CNN

This project implements a Convolutional Neural Network (CNN) for detecting spam SMS messages using PyTorch. The model is trained on the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) from Kaggle, achieving high accuracy in classifying messages as "spam" or "ham" (not spam).

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/BlackWyvernX/CNN_Spam_Classification.git
   cd Spam-detection
   ```

2. **Create a Conda Environment**:
   ```bash
   conda create --name learn --file requirements.txt
   conda activate learn
   ```

5. **Run the Script**:
   ```bash
   cd code
   python CNN.py
   ```

   ## Model Architecture
The CNN model consists of:
- **Embedding Layer**: 100-dimensional embeddings for a vocabulary of up to 10,000 words.
- **Convolutional Layers**: 128 filters with sizes [2, 3, 4] to capture n-grams.
- **Global Max Pooling**: Reduces feature maps to key features.
- **Dropout**: 0.5 to prevent overfitting.
- **Fully Connected Layer**: Outputs logits for 2 classes (spam, ham).
- **Loss Function**: Cross-entropy loss.
- **Optimizer**: Adam with learning rate 0.001.

## Dataset
The [Kaggle SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) contains:
- ~5,572 SMS messages.
- Labels: `ham` (not spam) or `spam`.
- ~13% spam messages, requiring class imbalance handling (e.g., weighted loss or oversampling).

## License

This project is licensed under the MIT License. The SMS Spam Collection Dataset is subject to its own licensing terms on Kaggle.

## Acknowledgments

- Dataset: [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- Built with PyTorch and NLTK

   
