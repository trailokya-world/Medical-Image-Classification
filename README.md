# 🧠 Brain Tumor MRI Classification

A deep learning project to classify brain tumors from MRI scans using CNN.

## 📌 Classes
- Glioma
- Meningioma
- No Tumor
- Pituitary

## 🗂️ Project Structure
```
project/
├── app.py                  # Streamlit web app
├── model_keras.keras       # Trained model
├── requirements.txt        # Dependencies
├── Brain_Tumor.ipynb       # Jupyter Notebook
└── README.md
```

## 🛠️ Tech Stack
- Python
- TensorFlow & Keras
- NumPy, Matplotlib
- Streamlit
- OpenCV / PIL

## 🏗️ Model Architecture
```
Input (224×224×3)
→ Data Augmentation
→ Conv2D(32) + BatchNorm + MaxPool
→ Conv2D(64) + BatchNorm + MaxPool
→ Conv2D(128) + BatchNorm + MaxPool
→ Conv2D(256) + BatchNorm + MaxPool
→ GlobalAveragePooling
→ Dense(128) + Dropout(0.4)
→ Dense(64)  + Dropout(0.3)
→ Dense(4, softmax)
```

## ⚙️ Installation

```bash
git clone https://github.com/your-username/brain-tumor-classification
cd brain-tumor-classification
pip install -r requirements.txt
```

## 🚀 Run Web App

```bash
streamlit run app.py
```

## 📊 Dataset
- **Source:** [Brain Tumor MRI Dataset - Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- Thousands of labeled MRI scans across 4 classes

## 📈 Results
| Metric | Score |
|--------|-------|
| Accuracy | 85–95% |
| Loss | Sparse Categorical Crossentropy |
| Optimizer | Adam |


