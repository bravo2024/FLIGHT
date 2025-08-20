
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.ticker as mtick
from amadeus import Client
import os
from sklearn.preprocessing import StandardScaler



client_id =st.secrets["AMADEUS_CLIENT_ID"]
client_secret =st.secrets["AMADEUS_CLIENT_SECRET"]
# Initialize Amadeus
try:
    amadeus = Client(client_id=client_id, client_secret=client_secret)
except Exception as e:
    st.error(f"Amadeus initialization failed: {e}")
    amadeus = None

# City to IATA mapping
#city_codes = {
 #   "Delhi": "DEL", "Mumbai": "BOM", "Bangalore": "BLR",
  #  "Chennai": "MAA", "Hyderabad": "HYD", "Kolkata": "CCU",
  #  "Goa": "GOI", "Ahmedabad": "AMD"
#}
# City to IATA mapping (expanded for India)
city_codes = {
    "Delhi": "DEL", "Mumbai": "BOM", "Bangalore": "BLR",
    "Chennai": "MAA", "Hyderabad": "HYD", "Kolkata": "CCU",
    "Goa (Dabolim)": "GOI", "Goa (Mopa)": "GOX", "Ahmedabad": "AMD",
    "Pune": "PNQ", "Jaipur": "JAI", "Lucknow": "LKO",
    "Amritsar": "ATQ", "Chandigarh": "IXC", "Guwahati": "GAU",
    "Patna": "PAT", "Indore": "IDR", "Bhopal": "BHO",
    "Nagpur": "NAG", "Varanasi": "VNS", "Ranchi": "IXR",
    "Bhubaneswar": "BBI", "Raipur": "RPR", "Surat": "STV",
    "Srinagar": "SXR", "Kochi": "COK", "Trivandrum": "TRV",
    "Madurai": "IXM", "Coimbatore": "CJB", "Mangalore": "IXE",
    "Visakhapatnam": "VTZ", "Agartala": "IXA", "Dehradun": "DED",
    "Port Blair": "IXZ", "Leh": "IXL", "Tirupati": "TIR"
}

# Duration calculation functions
def calculate_duration(dep_time_str, arr_time_str):
    dep_time = pd.to_datetime(dep_time_str)
    arr_time = pd.to_datetime(arr_time_str)
    # Handle next-day arrivals
    if arr_time < dep_time:
        arr_time += timedelta(days=1)
    duration_min = (arr_time - dep_time).total_seconds() / 60
    return duration_min

def format_duration(minutes):
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours}h {mins}m"

@st.cache_data(ttl=3600)
def get_flight_data(origin, destination, date):
    if not amadeus:
        return []
    try:
        response = amadeus.shopping.flight_offers_search.get(
            originLocationCode=origin,
            destinationLocationCode=destination,
            departureDate=date.strftime("%Y-%m-%d"),
            adults=1
        )
        results = []
        for flight in response.data:
            itinerary = flight["itineraries"][0]["segments"]
            dep_time = itinerary[0]["departure"]["at"]
            arr_time = itinerary[-1]["arrival"]["at"]
            airline = f"{itinerary[0]['carrierCode']} {itinerary[0]['number']}"
            price_inr = float(flight["price"]["total"]) * 83
            duration_min = calculate_duration(dep_time, arr_time)
            results.append({
                "Airline": airline,
                "Departure Time": dep_time,
                "Arrival Time": arr_time,
                "From": origin,
                "To": destination,
                "Price": price_inr,
                "Duration_Min": duration_min,
                "Duration": format_duration(duration_min),
                "Raw JSON": flight
            })
        return results
    except Exception as e:
        st.error(f"API error: {e}")
        return []

# Streamlit UI
st.set_page_config("VIVEK Flight Price Classifier", layout="wide")
st.title("✈️ VIVEK'S Flight Price Tracker & Classifier")

#tab1, tab2, tab3 = st.tabs(["🔍 Live Flights", "📈 Predict Price", "🧠 Classify Cheaper/Costlier"])
tab3, tab1,tab2 = st.tabs(["🧠 Classify Cheaper/Costlier","🔍 Live Flights","📈 Predict Price" ])
# --- Tab 1: Live Flights ---
with tab1:
    st.subheader("Search Flights")
    col1, col2, col3 = st.columns(3)
    with col1:
        from_city = st.selectbox("From", list(city_codes.keys()), key="from1")
    with col2:
        to_city = st.selectbox("To", list(city_codes.keys()), index=1, key="to1")
    with col3:
        flight_date = st.date_input("Departure Date", datetime.today(), key="date1")

    if st.button("Search Flights", key="search1"):
        flights = get_flight_data(city_codes[from_city], city_codes[to_city], flight_date)
        if flights:
            df = pd.DataFrame(flights)
            st.dataframe(df.drop(columns=["Raw JSON"]))

            st.markdown("#### 📊 Price Distribution")
            fig, ax = plt.subplots()
            sns.histplot(df["Price"], kde=True, bins=20)
            ax.set_xlabel("Price (INR)")
            ax.set_title("Distribution of Flight Prices")
            st.pyplot(fig)

            with st.expander("🧾 Raw Flight JSONs"):
                for f in df["Raw JSON"]:
                    st.json(f)
        else:
            st.warning("No flights found or error occurred.")

# --- Tab 2: Predict Price ---
with tab2:
    st.subheader("Flight Price Prediction (Linear Regression)")
    col1, col2, col3 = st.columns(3)
    with col1:
        from_city2 = st.selectbox("From", list(city_codes.keys()), key="from2")
    with col2:
        to_city2 = st.selectbox("To", list(city_codes.keys()), index=1, key="to2")
    with col3:
        flight_date2 = st.date_input("Departure Date", datetime.today(), key="date2")

    if st.button("Predict Prices", key="predict2"):
        flights2 = get_flight_data(city_codes[from_city2], city_codes[to_city2], flight_date2)
        if flights2:
            df = pd.DataFrame(flights2)
            df["Hour"] = pd.to_datetime(df["Departure Time"]).dt.hour
            df["Day"] = pd.to_datetime(df["Departure Time"]).dt.dayofweek
            df["AirlineCode"] = LabelEncoder().fit_transform(df["Airline"])
            df["Duration"] = df["Duration_Min"]

            features = ["Hour", "Day", "Duration", "AirlineCode"]
            X = df[features]
            y = df["Price"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

            model = LinearRegression()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            st.metric("Mean Absolute Error", f"{mean_absolute_error(y_test, predictions):.2f} INR")
            st.metric("R² Score", f"{r2_score(y_test, predictions):.3f}")

            fig2, ax2 = plt.subplots()
            sns.scatterplot(x=y_test, y=predictions, ax=ax2)
            ax2.set_xlabel("Actual Price")
            ax2.set_ylabel("Predicted Price")
            st.pyplot(fig2)

            coeff_df = pd.DataFrame({"Feature": features, "Weight": model.coef_})
            st.table(coeff_df)
        else:
            st.warning("No flights found.")

# --- Tab 3: Classification ---
with tab3:
    st.subheader("✈️ Flight Classification: Cheaper or Costlier")
    col1, col2, col3 = st.columns(3)
    with col1:
        from_city3 = st.selectbox("From", list(city_codes.keys()), index=list(city_codes.keys()).index("Kolkata"), key="from3")
    with col2:
        to_city3 = st.selectbox("To", list(city_codes.keys()), index=list(city_codes.keys()).index("Bangalore"), key="to3")
    with col3:
        flight_date3 = st.date_input("Departure Date", datetime.today() + timedelta(days=1), key="date3")

    future_days = st.slider("Select number of future days to Predict:", 1, 90, 30)

    if st.button("Classify Flights", key="classify3"):
        flights3 = get_flight_data(city_codes[from_city3], city_codes[to_city3], flight_date3)
        if flights3:
            df_class = pd.DataFrame(flights3)
            simulated_dfs = []
            for i in range(future_days):
                temp_df = df_class.copy()
                temp_df["Departure Time"] = pd.to_datetime(temp_df["Departure Time"]) + timedelta(days=i)
                temp_df["Arrival Time"] = pd.to_datetime(temp_df["Arrival Time"]) + timedelta(days=i)
                simulated_dfs.append(temp_df)

            df_class = pd.concat(simulated_dfs, ignore_index=True)

            df_class["Hour"] = pd.to_datetime(df_class["Departure Time"]).dt.hour
            df_class["Day"] = pd.to_datetime(df_class["Departure Time"]).dt.dayofweek
            df_class["Duration"] = df_class["Duration_Min"]

            # ✅ One-Hot Encode Airline
            df_class = pd.get_dummies(df_class, columns=["Airline"], drop_first=True)

            price_threshold = df_class["Price"].median()
            df_class["Class"] = (df_class["Price"] > price_threshold).astype(int)

            st.markdown(f"Flights above ₹{price_threshold:.2f} are classified as **Costlier**.")

            # Features after encoding
            features = ["Hour", "Day", "Duration"] + [col for col in df_class.columns if col.startswith("Airline_")]
            X_cls = df_class[features]
            y_cls = df_class["Class"]

            X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.25, random_state=42)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_cls)
            X_test_scaled = scaler.transform(X_test_cls)

            clf = LogisticRegression(C=0.1,penalty="l1",solver="liblinear",max_iter=1000)
            clf.fit(X_train_scaled, y_train_cls)
            y_pred_cls = clf.predict(X_test_scaled)
            y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]

            df_class["Duration_Display"] = df_class["Duration"].apply(lambda x: f"{int(x//60)}h {int(x%60)}m")

            test_results = df_class.iloc[y_test_cls.index].copy()
            test_results["Predicted"] = y_pred_cls
            test_results["Classification"] = test_results["Predicted"].map({0: "Cheaper", 1: "Costlier"})
            test_results["Flight Date"] = pd.to_datetime(test_results["Departure Time"]).dt.date
            test_results["Expected Class"] = test_results["Class"].map({1: "Costlier", 0: "Cheaper"})
            test_results["Correct"] = test_results["Classification"] == test_results["Expected Class"]
            incorrect = test_results[~test_results["Correct"]]

            # Metrics
            accuracy = accuracy_score(y_test_cls, y_pred_cls) * 100
            class_report = classification_report(y_test_cls, y_pred_cls, target_names=["Cheaper", "Costlier"], output_dict=True)
            cm = confusion_matrix(y_test_cls, y_pred_cls)
            logloss = log_loss(y_test_cls, clf.predict_proba(X_test_cls)
            fpr, tpr, _ = roc_curve(y_test_cls, y_pred_proba)
            auc_score = roc_auc_score(y_test_cls, y_pred_proba)

            st.markdown("### 📊 Performance Metrics")
            st.markdown(f"""
            - **Accuracy**: {accuracy:.2f}%
            - **Log Loss**: {logloss:.4f}
            - **AUC Score**: {auc_score:.3f}
            - **Precision (Cheaper)**: {class_report['Cheaper']['precision']:.3f}
            - **Recall (Cheaper)**: {class_report['Cheaper']['recall']:.3f}
            - **F1-Score (Cheaper)**: {class_report['Cheaper']['f1-score']:.3f}
            - **Precision (Costlier)**: {class_report['Costlier']['precision']:.3f}
            - **Recall (Costlier)**: {class_report['Costlier']['recall']:.3f}
            - **F1-Score (Costlier)**: {class_report['Costlier']['f1-score']:.3f}
            """)

            st.markdown("### 🧮 ROC Curve")
            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}", color="blue")
            ax_roc.plot([0, 1], [0, 1], "r--")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title("ROC Curve")
            ax_roc.legend()
            st.pyplot(fig_roc)

            st.markdown("### 🧮 Confusion Matrix")
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Cheaper", "Costlier"], yticklabels=["Cheaper", "Costlier"])
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            st.pyplot(fig_cm)


            st.markdown("### 📈 Price vs Classification")
            fig, ax = plt.subplots()
            cheaper = test_results[test_results["Classification"] == "Cheaper"]
            costlier = test_results[test_results["Classification"] == "Costlier"]
            ax.scatter(cheaper["Hour"], cheaper["Price"], color="green", label="Cheaper", alpha=0.6)
            ax.scatter(costlier["Hour"], costlier["Price"], color="red", label="Costlier", alpha=0.6)
            ax.set_xlabel("Departure Hour")
            ax.set_ylabel("Price (INR)")
            ax.set_title("Flight Price by Departure Hour")
            ax.legend()
            ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"₹{int(x):,}"))
            st.pyplot(fig)
        
        else:
            st.warning("No flights found.")
        
        
