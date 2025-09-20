import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------ Load Models ------------------------
scaler = joblib.load("rfm_scaler.pkl")       # StandardScaler
kmeans = joblib.load("kmeans_model.pkl")     # KMeans model

# ------------------------ Load and Prepare Product Matrix ------------------------
df = pd.read_csv("online_retail.csv")

product_matrix = pd.pivot_table(
    df,
    index="Description",
    columns="CustomerID",
    values="Quantity",
    aggfunc="sum",
    fill_value=0
)

similarity_matrix = cosine_similarity(product_matrix)
sim_df = pd.DataFrame(
    similarity_matrix,
    index=product_matrix.index,
    columns=product_matrix.index
)

# ------------------------ Streamlit App ------------------------
st.set_page_config(page_title="E-Commerce Intelligence Dashboard", layout="wide")

# ------------------------ Custom CSS ------------------------
st.markdown("""
<style>
 [data-testid="stAppViewContainer"] {
    background-color: #e0ebeb;  /* <-- your dashboard background color */
}
body {
    background-color: #ebfafa;
    color: #0f111a;
    font-family: 'Arial', sans-serif;
}
h1 {
    color: #75a3a3;
    font-weight: bold;
    text-align: center;
}
.stButton>button {
    background-color: #75a3a3;
    color: white;
    border-radius: 10px;
    font-size: 16px;
    height: 3em;
    width: 100%;
}
.card {
    background: linear-gradient(145deg, #ffffff, #e6f7f7);
    padding: 15px;
    margin: 10px 0;
    border-radius: 15px;
    border: 1px solid #ccc;
    box-shadow: 3px 3px 8px rgba(0,0,0,0.1);
    transition: transform 0.2s;
}
.card:hover {
    transform: translateY(-3px);
}
.similarity-bar {
    background:#e0e0e0;
    width:100%;
    border-radius:5px;
    margin-top:5px;
}
.similarity-fill {
    background:#75a3a3;
    padding:5px;
    border-radius:5px;
    color:#fff;
    text-align:center;
}
.metric-card {
    background: #ffffff;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-weight: bold;
    box-shadow: 3px 3px 8px rgba(0,0,0,0.1);
}
.metric-value {
    font-size: 28px;
    color: #75a3a3;
}
</style>
""", unsafe_allow_html=True)

# ------------------------ Dashboard Title ------------------------
st.markdown("<h1>🛒 Shopper Spectrum Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border:1px solid #ccc;'>", unsafe_allow_html=True)

# ------------------------ Metrics Row ------------------------
total_customers = df['CustomerID'].nunique()
total_products = df['Description'].nunique()
avg_purchase = df.groupby('CustomerID')['Quantity'].sum().mean()

col1, col2, col3 = st.columns(3)

# Total Customers Card
if col1.button(f"Total Customers\n{total_customers}", key="cust_card"):
    st.subheader("🧑‍🤝‍🧑 Top 5 Customers by Total Purchases")
    top_customers = df.groupby('CustomerID')['Quantity'].sum().sort_values(ascending=False).head(5)
    st.table(top_customers)

# Total Products Card
if col2.button(f"Total Products\n{total_products}", key="prod_card"):
    st.subheader("📦 Top 5 Best-Selling Products")
    top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(5)
    st.table(top_products)

# Avg Purchase per Customer Card
if col3.button(f"Avg Purchase\n{avg_purchase:.2f}", key="avg_card"):
    st.subheader("💰 Customers Above Average Purchase")
    avg_purchases = df.groupby('CustomerID')['Quantity'].sum()
    above_avg = avg_purchases[avg_purchases > avg_purchase].sort_values(ascending=False)
    st.table(above_avg.head(5))


# ------------------------ Tabs ------------------------
product_tab, segment_tab = st.tabs(["Product Recommendation", "Customer Segmentation"])

# ------------------------ 1. Product Recommendation ------------------------
with product_tab:
    st.subheader("🔍 Product Recommendation")
    st.write("Get 5 similar product suggestions based on what others bought.")

    product_input = st.selectbox(
        "Select a Product",
        sorted(product_matrix.index.unique()),
        help="Start typing to search for a product"
    )

    if st.button("Get Recommendations"):
        if product_input and product_input in sim_df.index:
            top_5_scores = sim_df[product_input].sort_values(ascending=False).iloc[1:6]

            # Vertical list for cards
            for i, (prod, score) in enumerate(top_5_scores.items(), 1):
                bar_width = int(score * 100)

                if i == 1:
                    badge_text, badge_color = "Top Pick", "#27AE60"
                elif i in [2,3]:
                    badge_text, badge_color = "Highly Similar", "#2980B9"
                else:
                    badge_text, badge_color = "Similar", "#F39C12"

                st.markdown(
                    f"""
                    <div class='card'>
                        <b>{i}. {prod}</b>
                        <span style='background-color:{badge_color}; color:white; 
                            padding:3px 8px; border-radius:8px; float:right;'>{badge_text}</span>
                        <div class='similarity-bar'>
                            <div class='similarity-fill' style='width:{bar_width}%;'>
                                Similarity: {score:.2f}
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.warning("Product not found. Try a different name.")

# ------------------------ 2. Customer Segmentation ------------------------
with segment_tab:
    st.subheader("📊 Customer Segmentation")
    st.write("Predict customer segment based on Recency, Frequency, and Monetary values.")

    rec = st.number_input("Recency (days since last purchase)", min_value=0)
    freq = st.number_input("Frequency (number of purchases)", min_value=0)
    mon = st.number_input("Monetary (total spend)", min_value=0.0)

    if st.button("Predict Cluster"):
        user_input = np.array([[rec, freq, mon]])
        scaled_input = scaler.transform(user_input)
        cluster = kmeans.predict(scaled_input)[0]

        label_map = {
            0: {"name": "At-Risk", "color": "#E74C3C"},
            1: {"name": "Occasional", "color": "#F39C12"},
            2: {"name": "Regular", "color": "#2980B9"},
            3: {"name": "High-Value", "color": "#27AE60"}
        }


        segment_info = label_map.get(cluster, {"name": "Unknown", "color": "#7F8C8D"})
        st.markdown(f"<div class='card' style='border-left: 5px solid {segment_info['color']}'>"
                    f"<h3>Predicted Customer Segment: {segment_info['name']} (Cluster {cluster})</h3>"
                    f"</div>", unsafe_allow_html=True)
