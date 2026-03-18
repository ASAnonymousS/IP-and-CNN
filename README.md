# IP and CNN

### A College Collaboration Project on Image Processing & CNNs

## 📌 Project Overview

This repository contains a collaborative effort to explore the intersection of classical **Image Processing** and **Convolutional Neural Networks (CNNs)**. Our goal is to develop a pipeline that cleans and augments raw image data before feeding it into a custom deep-learning model for classification/detection tasks.

### Core Objectives:

- Implement custom filters (Sobel, Gaussian, Laplacian) using NumPy/OpenCV.
- Develop a CNN architecture optimized for [mention dataset, e.g., CIFAR-10 or custom data].
- Compare performance between raw image inputs and pre-processed feature-mapped inputs.

------

## 🛠 Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow / Keras (or PyTorch)
- **Image Processing:** OpenCV, NumPy, Matplotlib

------

## 🚀 Getting Started

### Prerequisites

Ensure you have a C++ compiler and Python environment set up. If you are using a dedicated GPU ensure CUDA and cuDNN are correctly configured.

### Installation

1. **Clone the repository:**

   Bash (Using the https):

   ```bash
   git clone https://github.com/ASAnonymousS/IP-and-CNN.git
   cd IP-and-CNN
   ```

   Bash (Using the ssh): Recommended

   ```bash
   git clone git@github.com:ASAnonymousS/IP-and-CNN.git
   cd IP-and-CNN
   ```

2. **Create a virtual environment:**

   Linux:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

   Windows:

   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies:**

   Bash

   ```bash
   pip install -r requirements.txt
   ```

------

## 📂 Project Structure

Plaintext

```
.
├── References/
│   ├── Image_Samples/      # Standard test images (Lenna, Pout, etc.)
│   ├── MATLAB_Files/       # DSP & Signal Processing toolkit (.m scripts)
│   └── matplotlib/         # Python scripts for color space & histogram analysis
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
```

### 🛠 Reference Toolkit Usage

#### 1. MATLAB Setup

1. Navigate to: `cd References/MATLAB_Files`
2. Open scripts like `binaryimg.m` or `IMG01.m` in MATLAB to perform initial image manipulations.

#### 2. Matplotlib & Python Analysis

1. Navigate to: `cd References/matplotlib`
2. Run scripts for specific analysis:
   - `color_seperations.py`: Visualize RGB channels.
   - `histogram.py`: Generate pixel intensity distributions.
   - `rgb2gray_for_matplotlib.py`: Custom grayscale conversion logic.

---

## 🧪Ongoing Development

Our team is currently focused on mastering and implementing the following frameworks to build the end-to-end pipeline:

- [ ] **Data Visualization:** Learning **Matplotlib** for EDA and results plotting.
- [ ] **Numerical Computing:** Mastering **NumPy** for high-performance matrix operations.
- [ ] **Computer Vision:** Implementing image filtering and transformations with **OpenCV**.
- [ ] **Deep Learning:** Developing and training models using **TensorFlow**, **Keras**, and **PyTorch**.

------

## 🤝 Contributing

Since this is a collaboration, please follow the workflow to maintain project integrity:

1. **Pull** the latest changes from `main`.
2. **Create** a new branch for your feature (`git checkout -b your-name`).
3. **Environment Safety:** Ensure your local virtual environment is **not** tracked. If you create a new environment, verify that `.venv/` or `venv/` is listed in the `.gitignore` file before staging changes.
4. **Commit** your updates with clear, descriptive messages (and use GPG/SSH signing if configured).
5. **Push** and open a **Merge/Pull Request** with detailed explanation of the updates in the pull request.