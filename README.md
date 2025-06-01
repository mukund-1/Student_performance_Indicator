# Student Performance Prediction Web App

This is a Flask-based web application that predicts student performance based on various demographic and academic factors such as gender, parental education level, lunch type, and test preparation course. The prediction is performed using a trained machine learning pipeline.

---

## Features

- Input student details through a web form.
- Predicts student performance (e.g., math score or a performance category) using a trained model.
- Displays prediction result on the frontend.

---

## Project Structure

```
project/
│
├── app.py # Main Flask app
├── templates/
│ ├── index.html # Landing page
│ └── home.html # Input form and result display
├── src/
│ └── pipeline/
│ ├── predict_pipeline.py # CustomData and PredictPipeline classes
├── static/ # (Optional) CSS, JS, or image files
├── models/ # Saved model and preprocessing files
├── requirements.txt # Dependencies
└── README.md # Project description

```

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mukund-1/Student_performance_Indicator.git
cd student-performance-prediction
```

2. Create and activate a virtual environment (optional but recommended):

```bash 
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash

pip install -r requirements.txt
```
## Usage

Start the Flask application:

```bash

python app.py
```


## Notes
Ensure your trained model and any encoders or scalers are stored in the appropriate location and properly loaded by the PredictPipeline.

Update the predict_pipeline.py logic to reflect any changes in the model or input format.
