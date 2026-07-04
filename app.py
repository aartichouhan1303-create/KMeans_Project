import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="🌍 Country Clustering Dashboard",
    page_icon="🌍",
    layout="wide"
)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("Country-data.csv")
model = joblib.load("KMEANS_model.pkl")

# ---------------- CSS ----------------
st.markdown("""
<style>

.stApp{
background:linear-gradient(135deg,#0f172a,#1e3a8a,#2563eb);
color:white;
}

h1,h2,h3{
text-align:center;
color:white;
}

section[data-testid="stSidebar"]{
background:#111827;
}

[data-testid="stMetric"]{
background:#1f2937;
padding:15px;
border-radius:15px;
text-align:center;
}

.stButton>button{
background:#2563eb;
color:white;
border-radius:10px;
height:50px;
font-size:18px;
font-weight:bold;
width:100%;
}

.stButton>button:hover{
background:#1d4ed8;
}

.result{
background:#16a34a;
padding:20px;
border-radius:15px;
text-align:center;
font-size:25px;
font-weight:bold;
color:white;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🌍 Country Clustering Dashboard")

st.write(
"Analyze country development using the **KMeans Machine Learning Model**."
)

# ---------------- METRICS ----------------
c1,c2,c3,c4=st.columns(4)

c1.metric("🌍 Countries",len(df))
c2.metric("📊 Features",len(df.columns)-1)
c3.metric("🤖 Model","KMeans")
c4.metric("📈 Records",len(df))

st.divider()

# ---------------- SIDEBAR ----------------
st.sidebar.title("📂 Navigation")

page=st.sidebar.radio(
"Select Page",
[
"🏠 Dashboard",
"🤖 Prediction",
"📋 Dataset",
"📊 Visualizations",
"🔍 Country Search"
]
)

# ---------------- DASHBOARD ----------------
if page=="🏠 Dashboard":

    st.subheader("Income vs GDP")

    fig=px.scatter(
        df,
        x="income",
        y="gdpp",
        hover_name="country",
        color="income",
        size="health"
    )

    st.plotly_chart(fig,use_container_width=True)

# ---------------- PREDICTION ----------------
elif page == "🤖 Prediction":

    st.subheader("🤖 Predict Country Cluster")

    col1, col2 = st.columns(2)

    with col1:
        child_mort = st.number_input("Child Mortality", 0.0, 300.0, 20.0)
        exports = st.number_input("Exports", 0.0, 300.0, 40.0)
        health = st.number_input("Health Spending", 0.0, 20.0, 5.0)
        imports = st.number_input("Imports", 0.0, 300.0, 40.0)
        income = st.number_input("Income", 0.0, 200000.0, 5000.0)

    with col2:
        inflation = st.number_input("Inflation", 0.0, 100.0, 5.0)
        life_expec = st.number_input("Life Expectancy", 20.0, 100.0, 70.0)
        total_fer = st.number_input("Total Fertility", 0.0, 10.0, 2.5)
        gdpp = st.number_input("GDP Per Capita", 0.0, 100000.0, 10000.0)

    if st.button("🔍 Predict Cluster"):

        features = [[
            child_mort,
            exports,
            health,
            imports,
            income,
            inflation,
            life_expec,
            total_fer,
            gdpp
        ]]

        cluster = model.predict(features)[0]

        if cluster == 0:
            result = "🟢 Cluster 0"
        elif cluster == 1:
            result = "🟡 Cluster 1"
        else:
            result = "🔴 Cluster 2"

        st.success(f"Prediction Result : {result}")
# ---------------- DATASET ----------------
elif page=="📋 Dataset":

    st.subheader("Dataset Preview")

    st.dataframe(df,use_container_width=True)

    st.subheader("Dataset Statistics")

    st.dataframe(df.describe(),use_container_width=True)

# ---------------- VISUALIZATION ----------------
elif page=="📊 Visualizations":

    col1,col2=st.columns(2)

    with col1:

        fig1=px.histogram(
            df,
            x="income",
            nbins=20,
            title="Income Distribution"
        )

        st.plotly_chart(fig1,use_container_width=True)

    with col2:

        fig2=px.scatter(
            df,
            x="income",
            y="life_expec",
            color="health",
            hover_name="country",
            title="Income vs Life Expectancy"
        )

        st.plotly_chart(fig2,use_container_width=True)

    fig3=px.box(
        df,
        y="gdpp",
        title="GDP Distribution"
    )

    st.plotly_chart(fig3,use_container_width=True)

# ---------------- SEARCH ----------------
elif page=="🔍 Country Search":

    country=st.selectbox(
        "Select Country",
        sorted(df["country"])
    )

    result=df[df["country"]==country]

    st.dataframe(result,use_container_width=True)

    st.bar_chart(
        result.drop(columns=["country"]).T
    )

# ---------------- FOOTER ----------------
st.markdown("---")

st.markdown(
"""
<center>

### 🌍 Country Clustering Dashboard

Developed using ❤️ Python | Streamlit | Scikit-Learn | Plotly

</center>
""",
unsafe_allow_html=True
)