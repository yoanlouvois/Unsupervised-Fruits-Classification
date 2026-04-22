# Image Clustering Project (Fruits)
**ET4 Computer Science — Polytech Paris-Saclay**

## Description

The objective of this project is to compare different **feature engineering** methods and **unsupervised clustering** techniques applied to fruit images.

The main idea is to evaluate how the choice of descriptors influences the quality of the image grouping.

---

## Methodology

The project is based on two main stages:

### 1. Feature Extraction

Several approaches were tested:

* **Color Histograms**: Capture the distribution of colors in the image.
* **HOG (Histogram of Oriented Gradients)**: Capture shapes, contours, and textures.
* **Hu Moments**: Describe the shape of objects (invariant to transformations).
* **Pre-trained VGG16**: Deep feature extraction using a convolutional neural network.
* **Pre-trained ResNet50**: A more robust representation thanks to residual connections.

---

### 2. Clustering

Two unsupervised clustering methods were used:

* **K-Means**: Distance-based partitioning.
* **Spectral Clustering**: Based on the graph structure of the data.

---

### 3. Results Visualization

An interactive dashboard was developed using **Streamlit**, allowing you to:

* Compare results based on the chosen feature extraction method.
* Compare the performance of clustering algorithms.
* Visualize the generated image clusters.

---

## 4. Setup & Execution

### Step 1: Download Data and Install Packages
* Install the required dependencies:  
    `pip install -r requirements.txt`

### Step 2: Configure Data Path
* In the file `src/constant.py`, update the `PATH_DATA` variable with the path to the folder containing the images to be clustered.

### Step 3: Run the Clustering Pipeline
* Navigate to the `src` folder.
* Execute the following command:  
    `python pipeline.py`

### Step 4: Launch the Dashboard
* Navigate to the `src` folder.
* Execute the following command:  
    `streamlit run dashboard_clustering.py`
