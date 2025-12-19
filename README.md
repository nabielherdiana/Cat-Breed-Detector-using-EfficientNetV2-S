---

# 🐱 Cat Breed Detector using EfficientNetV2-S

An interactive **cat breed classification web application** built with **Streamlit** and powered by a **fine-tuned EfficientNetV2-S deep learning model**.
This project demonstrates the end-to-end process of **computer vision inference**, from image input to real-time prediction, deployed as a public web app.

🔗 **Live Demo:**
👉 [https://cat-breed-detector-using-efficientnetv2-s-lt4lwdkf4lwkf6mkkvnq.streamlit.app/](https://cat-breed-detector.streamlit.app/)

---

## ✨ Features

* Upload a cat image **from local file**
* Paste an **image URL** for instant prediction
* Predict **Top-3 cat breeds** with confidence scores
* Clean, modern, and user-friendly UI
* Deployed publicly using **Streamlit Community Cloud**

---

## 🧠 Model Overview

* **Architecture:** EfficientNetV2-S
* **Framework:** PyTorch
* **Technique:** Transfer Learning & Fine-Tuning
* **Task:** Multi-class image classification (cat breeds)
* **Output:** Top-3 predicted breeds with probabilities

The trained model is hosted externally on **Hugging Face Hub** and automatically downloaded when the app runs.

---

## 🖼️ How It Works

1. User uploads an image or pastes an image URL
2. Image is preprocessed (resize, normalization)
3. Image is passed through the EfficientNetV2-S model
4. The app displays the **Top-3 most likely cat breeds**

---

## 🚀 Tech Stack

* **Python**
* **PyTorch**
* **Torchvision**
* **Streamlit**
* **Pillow**
* **NumPy**
* **Hugging Face Hub** (model hosting)

---

## 📂 Project Structure

```
cat-breed-detector/
│
├── app.py              # Streamlit application
├── requirements.txt    # Dependencies
└── README.md           # Project documentation
```

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🌐 Deployment

This app is deployed using **Streamlit Community Cloud** and can be accessed publicly via the link below:

👉 [https://cat-breed-detector-using-efficientnetv2-s-lt4lwdkf4lwkf6mkkvnq.streamlit.app/](https://cat-breed-detector.streamlit.app/)

---

## ⚠️ Notes

* Predictions depend on image quality and visual clarity
* The model focuses on visual features and may confuse similar-looking breeds

---

## 📌 Author

**Nabiel Herdiana**
Statistics & Data Enthusiast | Machine Learning & Data Analytics

---

### ⭐ If you find this project interesting, feel free to give it a star!

---
