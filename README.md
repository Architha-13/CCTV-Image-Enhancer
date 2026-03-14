# CCTV-Image-Enhancer

# CCTV Image Enhancer using Real-ESRGAN

## Overview

This project enhances **low-resolution CCTV images** using a **Real-ESRGAN super-resolution model**.

The model is trained directly on **CCTV images** to reconstruct clearer high-resolution outputs from blurry or compressed surveillance frames.

Unlike traditional training pipelines, this implementation **trains directly on CCTV data without DIV2K pretraining**.

The trained model is then used inside a **Streamlit application** to enhance uploaded images.

---

# Features

* Super-resolution for low quality CCTV images
* Deep learning based enhancement using **Real-ESRGAN**
* Streamlit web interface for easy image upload
* Improves details in blurred or compressed surveillance frames

---

# Project Structure

```
CCTV-Image-Enhancer
│
├── app.py                 # Streamlit interface
├── sr_engine.py           # Super-resolution engine
├── requirements.txt       # Project dependencies
├── README.md
├── .gitignore
│
├── models/
│   └── net_g_15000.pth    # Trained model (download separately)
│
└── examples/
    ├── input.jpg
    └── output.jpg
```

---

# Model Details

Architecture:

```
RRDBNet
```

Generator parameters:

```
23 RRDB blocks
64 feature channels
scale factor = 4
```

Training configuration:

```
iterations: 15000
batch size: 4
scale: 4
patch size: 256
```

The final trained model checkpoint:

```
net_g_15000.pth
```

---

# Dataset

The model is trained using **CCTV image frames**.

Typical characteristics of the dataset:

* low resolution images
* motion blur
* compression artifacts
* low light conditions
* surveillance camera noise

Example dataset structure:

```
cctv_dataset/
   frame001.png
   frame002.png
   frame003.png
   frame004.png
```

Recommended dataset size:

```
5000 – 20000 images
```

---

# Installation

Clone the repository:

```
git clone https://github.com/Architha-13/CCTV-Image-Enhancer.git
cd CCTV-Image-Enhancer
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Download Model

The trained model is larger than GitHub's 100MB limit and is **not stored in the repository**.

Download the model file:

```
net_g_6000.pth from https://drive.google.com/drive/folders/1FYTDCjdMmA4EsnKx0QoVLsgicgFMo5WO?usp=sharing
```

Place it inside:

```
models/net_g_15000.pth
```

---

# Run the Application

Start the Streamlit interface:

```
streamlit run app.py
```

Upload a **low resolution CCTV image** and the system will generate a **4× enhanced output image**.

Example resolution improvement:

```
480×270  →  1920×1080
```

---

# Applications

Possible use cases include:

* CCTV footage enhancement
* license plate readability
* face detail reconstruction
* surveillance video restoration
* forensic analysis

---

# Technologies Used

* Python
* PyTorch
* Real-ESRGAN
* OpenCV
* Streamlit

---

# References

Real-ESRGAN Paper:

```
Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data
```

Official Repository:

```
https://github.com/xinntao/Real-ESRGAN
```

---

# Author

Architha
