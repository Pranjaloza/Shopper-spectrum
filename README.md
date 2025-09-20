# 🛒 Shopper Spectrum: Customer Segmentation & Product Recommendation Dashboard

## 📌 Project Overview
This project develops an **E-Commerce Customer Intelligence Dashboard** called **Shopper Spectrum**.  
It integrates **data preprocessing, RFM analysis, clustering, and product recommendation** into an interactive **Streamlit application**.  

The system enables businesses to:
- Segment customers into meaningful groups.
- Identify high-value, loyal, occasional, and at-risk customers.
- Recommend similar products to improve cross-selling and up-selling.
- Visualize trends in customer purchases and product sales.

---
## ⚙️ Features

### 🔹 Data Preprocessing & EDA
- Cleaned the dataset by handling **missing values** and **duplicates**.
- Conducted **exploratory data analysis (EDA)** to identify purchase patterns.
- Standardized **RFM (Recency, Frequency, Monetary)** values.

### 🔹 Customer Segmentation
- Applied **RFM analysis** for behavioral segmentation.
- Standardized RFM values using `StandardScaler`.
- Built **KMeans clustering model** to classify customers into segments:
  - 🟥 At-Risk  
  - 🟧 Occasional  
  - 🟦 Regular  
  - 🟩 High-Value  
- Visualized customer clusters using scatter plots.

### 🔹 Product Recommendation
- Constructed a **product-user matrix**.
- Computed **cosine similarity** between products.
- Recommended **top 5 similar products** for a given item.

### 🔹 Interactive Streamlit Dashboard
- Displays **business KPIs**: customers, products, purchases.
- Provides **product-based recommendations**.
- Predicts **customer segment** based on RFM values.
- Clean UI with customized **CSS styling** and light background.

---
## Model Details

- Scaler: **StandardScaler** applied to RFM values.
- Clustering: **KMeans** used for customer segmentation.
- Similarity: **Cosine similarity** for product recommendations.
- Saved Models:
  - **rfm_scaler.pkl** → For transforming new RFM inputs.
  - **kmeans_model.pkl** → For predicting customer clusters.
 
---
## 🖥️ Dashboard Workflow

🔹 **Home Dashboard**
- Displays overall customer and product metrics.
- Shows purchase trends and business KPIs.

🔹 **Product Recommendation Tab**
- Select a product → Get Top 5 most similar products with similarity scores.

🔹 **Customer Segmentation Tab**
- Enter Recency, Frequency, Monetary values.
- Model predicts the customer segment instantly.
 
  
