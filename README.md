# Multi-Modal-Fake-News-Detection
A multi-modal fake news detection model that uses BERT for text, ResNet for images, and R3D for videos to classify content as real or fake. It supports live inputs from Instagram, YouTube, and web articles with automatic content extraction and provides real-time predictions with accuracy metrics.
This model is designed to detect fake news using multi-modal inputs—specifically text, images, and video data—and can classify content as either real or fake. It is implemented in a Google Colab environment and supports live user input from URLs such as Instagram posts, YouTube links, and news/blog articles.

🔍 Key Features:

Textual Analysis using a pre-trained BERT model to understand the semantics of the claim or caption.
Image Understanding through a fine-tuned ResNet-18 backbone for contextual visual cues.
Video Understanding using a 3D CNN (R3D-18) model trained on the Kinetics-400 dataset to capture motion and temporal consistency.
Fusion Architecture that combines embeddings from all three modalities and uses a classification head to predict real vs. fake.
Dynamic Content Extraction from:
Instagram captions via Instaloader + fallback HTML scraping
News articles via newspaper3k
YouTube (planned for transcript support)
Robust Evaluation with sklearn's classification report (precision, recall, F1-score).
Live Inference Mode that accepts direct user input or links, extracts content, and performs prediction with instant feedback.
📊 Dataset Handling:

Accepts a CSV or dataframe format with:
text: Caption or article text
image_path: Local image path (optional)
video_path: Local video path (optional)
label: "real" or "fake"
Augments the dataset with additional annotated Instagram links for domain-specific accuracy improvement.
🛠 Training Details:

Loss Function: CrossEntropyLoss
Optimizer: Adam
Device: Supports both GPU and CPU
Model Checkpointing: Saves the best model as best_model.pt
