
# 🏡 Real Estate House Price Prediction

## 📌 Project Description

This project builds a **Real Estate House Price Prediction** system using Machine Learning.
The model predicts house prices based on real estate features such as location, house age, nearby facilities, and other relevant attributes.

The project includes:

* Data preprocessing and analysis
* Model training and saving
* A **Streamlit web application** for real-time price prediction

This project is developed as a **mini project** for academic learning and practical exposure to regression modeling and ML deployment.

---

## 📁 Dataset Information

* **Dataset Name:** Real Estate Dataset
* **File:** `Real estate.csv`

The dataset contains attributes such as:

* House age
* Distance to nearest MRT station
* Number of nearby convenience stores
* Latitude and longitude
* House price per unit area (target variable)

---

## 🛠️ Technologies & Libraries Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit

---

## 📂 Project Structure

```
Real_Estate_House_price
│
├── Real estate.csv
├── code.ipynb
├── app.py
├── models.pkl
├── scaler.pkl
├── requirements.txt
├── my_mlflow.py
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Selvaganapathy-k/Real_Estate_House_price
cd Real_Estate_House_price
```

---

### 2️⃣ (Optional) Create Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

---

### 3️⃣ Install Required Libraries

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run the Streamlit Application

```bash
streamlit run app.py
```

---

## 🌐 Live Application

🔗 **Streamlit App URL:**
[https://realestatehouseprice-ganapathy.streamlit.app/](https://realestatehouseprice-ganapathy.streamlit.app/)

---

## 🔍 Model Details

* Problem Type: **Regression**
* Trained model stored as: `models.pkl`
* Feature scaling handled using: `scaler.pkl`
* Predicts **house price per unit area**

---

## 📈 Features

* Clean and user-friendly Streamlit interface
* Real-time house price prediction
* Uses saved model and scaler for consistent results
* Easy to deploy and use

---

## 🎓 Learning Outcomes

* Understanding regression problems
* Data preprocessing and feature scaling
* Model saving and loading
* Building and deploying ML applications using Streamlit
* Structuring end-to-end ML projects on GitHub

---

## 📌 Notes

* Virtual environment folders (`venv`, `myvenv`) are not included in the repository.
* All required dependencies are listed in `requirements.txt`.

---

## ✍️ Author

**Selvaganapathy K**
Computer Science Student

---

## 🏁 Conclusion

This project demonstrates how machine learning can be applied to real estate data to predict house prices and provide a practical, real-time prediction system using Streamlit.
