# TrafficTelligence-Advanced-Traffic-Volume-Estimation-with-Machine-Learning
## Project Overview

This project is a web-based application that estimates **traffic volume** at a given time and location using **Machine Learning**. It leverages historical traffic and environmental data (like weather, holidays, time) to make predictions. The system is built using a **Flask backend** and a trained machine learning model, providing a simple interface for users to input conditions and receive an estimated vehicle count.

---

## Problem Statement

With the rapid growth of urban areas, managing road traffic efficiently is a critical need. Predicting traffic volume accurately helps:
- City planners optimize infrastructure.
- Traffic departments manage congestion.
- Ride-sharing services improve routing.

Manual counting and fixed-schedule estimations are often inaccurate. This project uses **data-driven predictions** to solve the problem.

---

## Step-by-Step process of the project

### 1. Collected and Preprocessed Data
- Used a dataset with features like:
  - `holiday`, `weather`
  - `temperature`, `humidity`
  - `time`, `date`, and derived features like `hour`
- Categorical data was encoded.
- Continuous data was scaled using `StandardScaler`.

### 2. Trained a Machine Learning Model
- Selected ML algorithms (e.g., Random Forest).
- Performed train-test split.
- Trained the model and saved it using `joblib` as `model.pkl`.
- Also saved:
  - `scaler.pkl` for consistent input scaling.
  - `columns.pkl` to track one-hot encoded columns.

### 3. Built a Flask Web Application
- Created a Flask backend in `app.py`.
- Designed an HTML form (`index.html`) to accept inputs.
- Used `traffic.py` to load models and make predictions.
- Displayed the result in `result.html`.

### 4. Created Web Pages
- `index.html`: Input form
- `result.html`: Shows predicted traffic volume
- Used `static/` for background images.

---

##  Screenshots
###  Correlation Heatmap 
<img width="1787" height="952" alt="Correlation Heatmap img" src="https://github.com/user-attachments/assets/7ee389f3-8f3c-4612-a475-14b58a4dac01" />

### Pair plot 
<img width="1906" height="946" alt="Pair plot img" src="https://github.com/user-attachments/assets/b7144951-d8d5-4858-9502-6745e205e27c" />

###  Input Form 
<img width="1827" height="891" alt="input form img" src="https://github.com/user-attachments/assets/25f1a469-fbf2-4b09-9a5d-fa940af950c0" />

###  Prediction Result
<img width="1643" height="804" alt="prediction result img" src="https://github.com/user-attachments/assets/759a9f7a-4660-49ff-a49e-c8699a207ca5" />


---

##  Files & Folder Structure

```
Traffic_volume_estimation/
├── app.py
├── traffic.py
├── model.pkl
├── scaler.pkl
├── columns.pkl
├── requirements.txt
├── static/
│   └── image1.jpg, image2.jpg
└── templates/
    ├── index.html
    └── result.html
```

---

##  How to Use the App

1. Run the app:
   ```bash
   python app.py
   ```
2. Open in browser:
   ```
   http://127.0.0.1:5000
   ```
3. Fill the form → Submit → View prediction

---

##  Key Features

- ML-based prediction
- Real-time input/output
- Lightweight Flask app
- Clean UI with HTML/CSS
- Model + preprocessing saved for deployment

---

## Technologies Used

| Area | Tools |
|------|-------|
| Backend | Flask |
| ML | scikit-learn, pandas |
| Frontend | HTML, CSS |
| Model Storage | joblib |

---

##  Learning Outcomes

- End-to-end ML workflow (preprocess → train → deploy)
- One-hot encoding, scaling, joblib
- Flask integration with HTML
- Input handling and prediction on the web

---
#####  Download Files

You can download the trained model file from Google Drive:

👉 [Click here to download] (https://drive.google.com/drive/folders/1sLDuugX8bWk-wvYR925ZKTRFOqW6AxVl?usp=drive_link)



