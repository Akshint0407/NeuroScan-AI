# 🧠 NeuroScan AI - Brain Tumor Detection from MRI Scans

![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

A deep learning-powered diagnostic tool that analyzes MRI scans to detect potential brain tumors using convolutional neural networks (CNNs).

## ✨ Features

- 🖼️ Upload MRI scans in common image formats (JPG, PNG, JPEG)
- 🔍 AI-powered tumor detection with confidence percentage
- 📊 Visual results with color-coded diagnosis
- 📝 Detailed recommendations based on findings
- 📱 Responsive design works on desktop and mobile

## 🛠️ Installation

1. Clone the repository:
```bash
  https://github.com/Akshint0407/NeuroScan-AI.git
```
```
  cd NeuroScan-AI
```
2. Install dependencies:

```bash
pip install -r requirements.txt
```
3. Run the Streamlit app:

```bash
streamlit run app.py
```

## 🧩 Project Structure

| File/Folder               | Description                          |
|---------------------------|--------------------------------------|
| `app.py`                  | Streamlit application main file      |
| `style.css`               | Custom styling for the web app       |
| `brain_tumor_model.keras` | Trained CNN model for tumor detection|
| `requirements.txt`        | Python dependencies list             |
| `README.md`               | Project documentation                |

## 📂 Dataset
The model was trained on the following datasets:

- Brain Tumor MRI Dataset from kaggle

## 🏗️ Model Architecture
The CNN model consists of:

- 4 Convolutional layers with ReLU activation

- MaxPooling layers for dimensionality reduction

- Dropout layers for regularization

- Dense layers for classification

- Achieved 92.68% validation accuracy on test data.

## 👨‍💻 Developers
- Akshint
- Dhruv

## 📜 License
This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request.

***Disclaimer: This tool is for educational and research purposes only. Always consult a medical professional for clinical diagnosis.***
