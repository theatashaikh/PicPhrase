## Image Captioning with Deep Learning

This project implements a deep learning model for generating captions of images.

App is live on hugging face space: https://huggingface.co/spaces/theatashaikh/PicPhrase

**Model Architecture:**

- **Feature Extraction:** InceptionNetv3 pre-trained model is used to extract image features.
- **Caption Generation:** Long Short-Term Memory (LSTM) network is used to generate captions based on the extracted features.

**Dataset:**

- Flickr30k: This dataset consists of over 31,000 images with 5 captions each, totaling over 155,000 captions.

**Implementation Details:**

- Framework: TensorFlow
- Word Embeddings: fastText pre-trained word vectors

**Experiments:**

- Various pre-trained CNN models (ResNet50, EfficientNetB0, VGG16) were evaluated for feature extraction. InceptionNetv3 achieved the best performance.
- Training on flickr8k (8,000 images with captions) resulted in overfitting. Flickr30k was chosen to address this issue.

**Training:**

- Platform: Google Colab
- Hardware: T4 GPU (free tier)
- Training Time: 20 epochs (~2 hours)

**Results:**

- Training Loss: 4.3254 -> 0.9670
- Training Accuracy: 0.2456 -> 0.7009
- Validation Loss: 5.7075 -> 5.4508
- Validation Accuracy: 0.1935 -> 0.2816
- BLEU Score: BLEU-1: 0.538838, BLEU-2: 0.305489

**Further Exploration:**

- Experiment with different LSTM architectures (e.g., attention mechanism)
- Explore alternative datasets for captioning tasks.
