
# 🖼️ Image Denoising

## 📌 Overview
Image denoising is a process of removing noise from images while preserving important details. This project implements **image denoising** using advanced **Deep Learning** and **Computer Vision** techniques.

---

## ✨ Features
- **Removes noise** from images using deep learning models
- **Preserves essential details** while reducing unwanted artifacts
- **Supports multiple denoising techniques** (Gaussian Blur, Non-Local Means, Deep Learning models)
- **User-friendly interface** to upload and process images
- **Batch processing** for multiple images

---

## 📊 Methods Used
### 🔹 Traditional Methods:
- **Gaussian Blur**: Smooths out noise by averaging neighboring pixels.
- **Median Filtering**: Replaces each pixel with the median value of neighboring pixels.
- **Non-Local Means Denoising**: Uses similar patches within the image for noise reduction.

### 🔹 Deep Learning Methods:
- **Autoencoders**: A neural network-based approach that learns to reconstruct noise-free images.
- **Denoising CNNs**: Pre-trained convolutional neural networks designed to remove noise.
- **Denoising Diffusion Probabilistic Models (DDPMs)**: Advanced deep learning-based image restoration models.

---

## 📷 Sample Results
| Noisy Image | Denoised Image |
|-------------|---------------|
| ![Noisy](photo/noisy.png) | ![Denoised](photo/denoised.png) |

---

## 🛠️ Installation & Setup

### 🔹 Clone the Repository:
```sh
git clone https://github.com/rounakkumar30/Image-Denoising.git
cd Image-Denoising
```

### 🔹 Install Dependencies:
```sh
pip install -r requirements.txt
```

### 🔹 Run the Application:
```sh
python app.py
```

---

## 📚 Technologies Used
- **Python** (OpenCV, NumPy, Scikit-Image)
- **TensorFlow / PyTorch** (For deep learning models)
- **Flask / Django** (For web-based implementation)
- **D3.js / Matplotlib** (For visualizing denoising results)

---

## 🤝 Contribution
Contributions are welcome! Feel free to submit **pull requests** or open an **issue**.

---

## 📜 License
This project is licensed under the **MIT License**.

---

🚀 **Developed by [Rounak Kumar](https://github.com/rounakkumar30)**
```
