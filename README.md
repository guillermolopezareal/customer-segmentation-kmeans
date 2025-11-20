# 🛒 Customer Segmentation with K-Means (Online Retail Dataset)

This repository contains an end-to-end customer segmentation analysis using **unsupervised learning**, applied to the **Online Retail dataset** (UCI ML Repository). The project extracts behavioral features from transaction data and applies **K-Means clustering** to identify meaningful customer groups.

The full implementation is available in the notebook:

📄 `online-retail-data-clustering.ipynb`

---

## 🚀 Project Overview

Understanding customer behavior is critical for:

- targeted marketing  
- personalized communication  
- product recommendation strategies  
- churn prediction  
- revenue optimization  

This project builds an analytical pipeline that:

1. Cleans and filters retail transaction data  
2. Extracts customer-level behavioral features  
3. Applies **RFM-inspired feature engineering** (Recency, Frequency, Monetary value)  
4. Scales and normalizes the dataset  
5. Uses **K-Means** to cluster customers  
6. Evaluates clusters with:
   - inertia (SSE)
   - elbow method  
   - silhouette score  
7. Visualizes final clusters  
8. Interprets customer segments

---

🧠 Feature Engineering

The notebook calculates customer-level metrics including:
RFM-style behavioral features
  1. Recency — days since last purchase
  2. Frequency — number of invoices
  3. Monetary Value — total spending
     
Additional features
  1. average order value
  2. number of items purchased
  3. cancellations
  4. geographic region
These features form the basis for clustering.

🔧 Methodology

**Data Cleaning**
  1. Remove missing Customer IDs
  2. Filter returns/cancellations
  3. Remove negative quantities
  4. Handle duplicates
  5. Convert datatypes (InvoiceDate → datetime)

**2. Aggregation**

Each customer becomes one row using:
aggregated_df = cleaned_df.groupby("Customer ID").agg(
    MonetaryValue=("SalesLineTotal", "sum"),
    Frequency=("Invoice", "nunique"),
    LastInvoiceDate=("InvoiceDate", "max")
)

**3. Scaling**
StandardScaler
Log-transform or MinMaxScaler if needed

**4. Clustering**

Fit K-Means for k = 2–10
Evaluate with:
Elbow method
Silhouette score
Select optimal k

**5. Visualization**

PCA 2D projection
Cluster scatterplots
Centroid comparison
Feature distribution by cluster

📊 Results
Example outputs included in the notebook:
