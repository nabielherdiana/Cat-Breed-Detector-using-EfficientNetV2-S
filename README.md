# Cat Breed Detector using EfficientNetV2-S

An interactive **cat breed classification web application** built with **Streamlit** and powered by a **fine-tuned EfficientNetV2-S deep learning model**.
This project demonstrates the end-to-end process of **computer vision inference**, from model development to real-time deployment as a public web application.

**Live Demo:**
[Cat Breed Detector Web Application](https://cat-breed-detector.streamlit.app/)

### Features

* Upload a cat image **from local file**
* Paste an **image URL** for instant prediction
* Predict **Top-3 cat breeds** with confidence scores
* Clean, modern, and user-friendly UI
* Deployed publicly using **Streamlit Community Cloud**

### Model Overview

* **Architecture:** EfficientNetV2-S
* **Framework:** PyTorch
* **Technique:** Transfer Learning & Fine-Tuning
* **Task:** Multi-class image classification (cat breeds)
* **Output:** Top-3 predicted breeds with probabilities

The trained model is hosted externally on **Hugging Face Hub** and automatically downloaded when the app runs.

### Model Development (Google Colab)

The complete **model training and experimentation pipeline** was developed using **Python in Google Colab**, including:

* Data loading and preprocessing
* Image augmentation
* Transfer learning with EfficientNetV2-S
* Model training and validation
* Performance evaluation
* Model saving for deployment

**Google Colab Notebook (Modeling & Training):**
[Cat Breed Classification with CNN](https://colab.research.google.com/drive/1AwfTfBSdPDQI-DJv8l1m9WIJxmTJP5YQ)

This notebook demonstrates the **machine learning workflow behind the deployed model**, complementing the Streamlit inference application.

### How It Works

1. User uploads an image or pastes an image URL
2. Image is preprocessed (resize, normalization)
3. Image is passed through the EfficientNetV2-S model
4. The app displays the **Top-3 most likely cat breeds**

### Sample Predictions

### Prediction Gallery

To evaluate model performance qualitatively, we visualized predictions on a random batch of test images. The grid below displays the **Actual Class**, the **Predicted Class**, and the **Confidence Score**.

![Sample Predictions](assets/cat-prediction.png)

*Figure: Random batch of inference results. While the model performs well on distinctive breeds (e.g., Sphynx, Siamese), it faces challenges with visually similar breeds, such as distinguishing between **Maine Coon** and **Norwegian Forest**, which is a known challenge in fine-grained classification.*

### Tech Stack

* **Python**
* **PyTorch**
* **Torchvision**
* **Streamlit**
* **Pillow**
* **NumPy**
* **Hugging Face Hub** (model hosting)
* **Google Colab** (model training & experimentation)

### Project Structure

```
cat-breed-detector/
│
├── app.py              # Streamlit application (inference)
├── requirements.txt    # Dependencies
└── README.md           # Project documentation
```

### Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Deployment

This app is deployed using **Streamlit Community Cloud** and can be accessed publicly:

[https://cat-breed-detector.streamlit.app/](https://cat-breed-detector.streamlit.app/)

### Notes

* Predictions depend on image quality and visual clarity
* The model focuses on visual features and may confuse visually similar breeds

### Author

- Nabiel Alfallah Herdiana  
- Rasyid Abdurrahman  
- Yaafi Ferdian Syahputra  
- Farhan Alkarimi

---

If you find this project interesting, feel free to give it a star⭐!

---
