# Customer Segmentation Using K-Means Clustering

-> Project Overview
This project uses Machine Learning techniques to segment customers into different groups based on their characteristics and purchasing behaviour.

The project uses the K-Means clustering algorithm and provides an interactive interface through Streamlit.

-> Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

-> Project Files

File                         Description               

 `app.py`                     Streamlit application     
 `kmeans_model.pkl`           Trained K-Means model     
 `scaler.pkl`                 Fitted feature scaler     
 `customer_segmentation.csv`  Customer dataset          
 `requirements.txt`           Required Python libraries 

->  Installation

1. Clone this repository:
```
git clone YOUR_REPOSITORY_URL
```

2. Navigate to the project folder:
```
cd Customer-Segmentation
```

3. Install the dependencies:
```
pip install -r requirements.txt
```

-> Running the Application

Run the following command:
```
streamlit run app.py
```

The application will open in your browser at:

`http://localhost:8501`

->  Machine Learning Algorithm

K-Means Clustering is used to group customers based on similarities in their features.

The trained model and fitted scaler are saved using Joblib and loaded by the Streamlit application.

-> Author

Developed as a Customer Segmentation Machine Learning project.
