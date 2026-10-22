# Alzheimer's Disease Detection from Brain MRI

![Python](https://img.shields.io/badge/Python-3-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Flask](https://img.shields.io/badge/Web-Flask-lightgrey)
![Status](https://img.shields.io/badge/status-academic%20prototype-yellow)

A deep learning project that looks at a **brain MRI scan** and predicts which **stage of dementia** it shows. A convolutional neural network (CNN) does the classification, and a small **Flask web app** lets anyone upload a scan and see the result in the browser.

B.Tech final-year project, RGUKT Basar (2023–2024).

> ⚠️ **Research and teaching prototype only.** This is not a medical device and has not been validated for clinical use. Do not use it to diagnose anyone.

---

## Contents
- [The problem](#the-problem)
- [What the app does](#what-the-app-does)
- [How it works](#how-it-works)
- [Demo](#demo)
- [Repository structure](#repository-structure)
- [Testing](#testing)
- [Tech stack](#tech-stack)
- [Limitations](#limitations)
- [Team](#team)
- [References](#references)

---

## The problem
Alzheimer's disease is the most common form of dementia and has no cure, so **catching it early** matters.

Earlier approaches we reviewed used **tabular patient data** (age, number of visits, and so on) with classical models such as SVM and Random Forest. That data is noisy and hand-crafting features from it is slow.

**Our approach:** skip the tabular data and classify the **MRI image directly** with a CNN, which learns the useful features on its own.

## What the app does
1. You upload a brain MRI image (`.jpg`) in the browser.
2. The image is resized to **176 × 176** pixels.
3. A trained Keras model predicts one of **four classes**:

| Class | Meaning |
|---|---|
| No Dementia | No signs of dementia detected |
| Very Mild Dementia | Earliest detectable stage |
| Mild Dementia | Early stage |
| Moderate Dementia | More advanced stage |

4. The predicted stage is shown on the result page.

## How it works

### The model
A 5-layer convolutional neural network built with TensorFlow / Keras:

```
Input (176×176 MRI) → Convolution → Max Pooling → Dense → Output (4 classes)
```

![CNN model diagram](docs/cnn-model-diagram.png)

The project report also discusses the **VGG-16** architecture alongside this CNN.

### The web app
The Flask upload route:
1. saves the uploaded file safely (`secure_filename`),
2. loads and resizes it to 176 × 176,
3. runs it through the model,
4. picks the class with the highest score and maps it to a stage name.

Excerpt from the report:

```python
def upload_image():
    file = request.files['file']
    file.save(os.path.join("static/uploads", secure_filename(file.filename)))

    imvar = tf.keras.preprocessing.image.load_img(
        os.path.join("static/uploads", secure_filename(file.filename))).resize((176, 176))
    imarr = tf.keras.preprocessing.image.img_to_array(imvar)
    imarr = np.array([imarr])
    model = tf.keras.models.load_model("model")
    impred = model.predict(imarr)

    def roundoff(arr):
        """Round off according to the argmax of each predicted label array."""
        arr[np.argwhere(arr != arr.max())] = 0
        arr[np.argwhere(arr == arr.max())] = 1
        return arr
    ...
    classdict = {1: "Mild Dementia", 2: "Moderate Dementia",
                 3: "No Dementia, Patient is Safe", 4: "Very Mild Dementia"}
```

## Demo

**1. Upload a scan**

| Home page | File selected |
|---|---|
| ![home](docs/screenshots/ui-home.jpg) | ![upload](docs/screenshots/ui-upload-with-file.jpg) |

**2. See the prediction**

| No dementia | Very mild dementia |
|---|---|
| ![none](docs/screenshots/result-no-dementia.jpg) | ![very mild](docs/screenshots/result-very-mild-dementia.jpg) |

| Mild dementia | Moderate dementia |
|---|---|
| ![mild](docs/screenshots/result-mild-dementia.jpg) | ![moderate](docs/screenshots/result-moderate-dementia.jpg) |

## Repository structure

```
alzheimers-mri-detection/
├── README.md
└── docs/
    ├── cnn-model-diagram.png     # CNN architecture
    ├── screenshots/              # Web app screenshots (upload + 4 results)
    └── uml/                      # Design diagrams
        ├── use-case.png
        ├── class-diagram.png
        ├── activity-diagram.png
        └── sequence-diagram.png
```

> This repository currently holds the **documentation, design diagrams and screenshots** of the project. The trained model, dataset and training notebook are not included.

## Testing
Two functional tests on the upload flow, both **passed**:

| # | Test | Expected result |
|---|---|---|
| 1 | Image format | A valid `.jpg` is accepted and a result is shown; other formats are rejected |
| 2 | Image size | A valid-size image is processed; an invalid size asks the user for a correct image |

## Tech stack
- **Language:** Python
- **Deep learning:** TensorFlow, Keras, NumPy
- **Web app:** Flask, HTML, CSS, JavaScript
- **Design:** UML (use case, class, activity, sequence diagrams in [`docs/uml`](docs/uml))

## Limitations
- Not tested on clinical data, other MRI scanners or other populations.
- Classifies single 2D slices only: no 3D context and no patient history.
- Trained model, dataset and training code are not versioned in this repository.

## Team
Final-year project by **T. Sreya**, **A. Sai Ruthik** and **T. Falguni**.

- **Guide:** Mr. A. Surender, Assistant Professor
- **Coordinator:** Mrs. Latha Reddy, Assistant Professor

## References
1. M. J. Knight, B. McCann, R. A. Kauppinen, E. J. Coulthard, *Magnetic Resonance Imaging to Detect Early Molecular and Cellular Changes in Alzheimer's Disease*.
2. K. Trojachanec, I. Kitanovski, I. Dimitrovski, S. Loshkovska, *Longitudinal brain MRI retrieval for Alzheimer's disease using different temporal information*.
3. S. Sarraf, G. Toghi, *Classification of Alzheimer's disease using fMRI data and deep learning convolutional neural networks*.
4. N. Bhagwat, *Prognostic applications for Alzheimer's disease using magnetic resonance imaging and machine learning*.
5. J. Shijie, W. Ping, J. Peiyi, H. Siping, *Research on data augmentation for image classification based on convolution neural networks*.
6. S. Al-Shoukry, T. H. Rassem, N. M. Makbol, *Alzheimer's Diseases Detection by Using Deep Learning Algorithms: A Mini-Review*, IEEE Access, 2020.
