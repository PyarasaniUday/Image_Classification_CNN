# 🧠 Image Classification using CNN (CIFAR-10)

## 📌 Project Overview

This project is a **CNN-based Image Classifier** that can identify objects in images.
It classifies images into **10 categories** like car, dog, cat, airplane, etc.

---

## 🎯 Aim of the Project

* Identify objects in images
* Classify images into categories
* Provide prediction with confidence score

---

## 🧠 What is CNN?

CNN (Convolutional Neural Network) is a deep learning model used for **image processing**.

👉 Simple idea:

* Human sees image → understands object
* CNN sees pixels → learns patterns → predicts object

---

## 🧩 CNN Structure

* **Input Layer** → Image (32×32×3)
* **Convolution Layers** → detect edges, shapes, textures
* **Pooling Layers** → reduce size, keep important features
* **Flatten Layer** → convert to 1D
* **Dense Layer** → classification
* **Output Layer (Softmax)** → gives probabilities

---

## 📂 Dataset

We used **CIFAR-10 dataset**, which contains:

* 60,000 images
* 10 classes:

  * airplane
  * car
  * bird
  * cat
  * deer
  * dog
  * frog
  * horse
  * ship
  * truck

---

## 🔄 Project Workflow

1. **Data Collection**

   * Loaded CIFAR-10 dataset

2. **Data Preprocessing**

   * Normalize pixel values (0–1)
   * Resize images (32×32)

3. **Model Building**

   * CNN model using:

     ```
     Conv → Pool → Conv → Pool → Flatten → Dense
     ```

4. **Training**

   * Optimizer: Adam
   * Loss: Cross-Entropy

5. **Evaluation**

   * Tested model accuracy (~70%)

6. **Model Saving**

   * Saved as:

     ```
     models/cifar_cnn_model.h5
     ```

7. **Prediction**

   * Predicts class for new images

8. **Deployment**

   * Built UI using Streamlit

---

## 🤖 How Prediction Works

1. Image is resized to 32×32
2. Converted into pixel values
3. CNN extracts features
4. Softmax gives probabilities

Example:

```
car → 0.85
truck → 0.10
ship → 0.05
```

Final Output:

```
Prediction: car
Confidence: 0.85
```

---

## 💻 How to Run the Project

### 1️⃣ Install Requirements

```
pip install -r requirements.txt
```

### 2️⃣ Run the App

```
streamlit run app_ui.py
```

---

## 📸 Features

* Upload image
* Get prediction instantly
* Shows confidence score
* Displays top predictions

---

## 🏆 Conclusion

This project demonstrates how CNN can be used to **classify images effectively**.
It also shows how to build a **complete ML pipeline from training to deployment**.

---

## 🚀 Future Improvements

* Improve accuracy using deeper CNN
* Add more classes
* Use real-time camera input
* Add heatmap (Grad-CAM) visualization

---

## 👨‍💻 Author

Pyarsani Uday Kumar
