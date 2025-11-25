MRI Brain Tumor Detection & Segmentation

Automated Web System using **UNet**, **GrabCut**, **MaxFlow**, and **Random Forest**.

All the required libraries are in the req.txt file just run (pip install -r req.txt)

Project Overview
This project performs **MRI brain tumor segmentation & classification** using:
- UNet (deep learning segmentation)
- GrabCut & MaxFlow (graph-based segmentation)
- Feature extraction using GLCM + RegionProps
- Random Forest classifier
- Streamlit Web App interface

---

Project Structure
```

mri-tumor-detection
│
├── Tumor Classification.ipynb  → ML training (features + Random Forest)
├── BRISC_UNet.pth              → Trained UNet model (for segmentation)
├── tumor_classifier_model.pkl  → Saved Random Forest classifier
├── web.py (Streamlit code) → Main Web Application
├── Model Train.ipynb           → UNet Training Notebook
├── OUTPUT.ipynb                → Post-processing + Segmentation Notebook
└── README.md                   → Documentation (this file)

```


Workflow Pipeline
```
Upload MRI
     ↓
UNet → Probability Map
     ↓
GrabCut / MaxFlow Refinement
     ↓
Feature Extraction (8 features)
     ↓
Random Forest Classifier
     ↓
Streamlit Web App → Final Output

```

---

Feature Extraction (8 Features)
| Type           | Features |
| Shape          | Area, Perimeter, Eccentricity, Solidity |
| Texture (GLCM) | Contrast, Energy, Homogeneity, Correlation |

---

Run the Web App
```bash
streamlit run web_app.py
```
 

Segmentation Models
| Model | Type | Output |
|-------|------|--------|
| UNet | Deep Learning | Probability Map |
| GrabCut | Graph + GMM | Binary Mask |
| MaxFlow | Min-Cut Optimization | Binary Mask |

---

Libraries Used
| Category           | Libraries                                        |
| ------------------ | ------------------------------------------------ |
| Deep Learning      | PyTorch, UNet                                    |
| Segmentation       | OpenCV GrabCut, MaxFlow                          |
| Feature Extraction | `skimage.measure`, `graycomatrix`, `graycoprops` |
| ML Classification  | RandomForest (Scikit-Learn)                      |
| Frontend           | Streamlit                                        |
| Utils              | NumPy, Matplotlib, PIL                           |
 


Segmentation Techniques
1) UNet (Deep Learning Based)

Pixel-wise segmentation

Output: Probability Map (0–1)

Trained on BRATS dataset

2) GrabCut (Graph-Based Segmentation)

Uses GMM + Iterative GraphCut

Works better when seed mask is available

Uses pixel probability from UNET

3) MaxFlow / MinCut

Creates cost function from -log(probability)

Pixel grouping via graph optimization

Final output = binary mask (0 or 255)

#Feature Extraction (8 Features)

From segmented tumor region using regionprops() + GLCM:

| Type           | Feature Name                               |
| -------------- | ------------------------------------------ |
| Shape          | Area, Perimeter, Eccentricity, Solidity    |
| Texture (GLCM) | Contrast, Energy, Homogeneity, Correlation |

These are used to train Random Forest classifier.

Output Categories
✔ Glioma 
✔ Meningioma 
✔ Pituitary Tumor 
✔ No Tumor 


