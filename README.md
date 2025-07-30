# ✈️ Dynamic Flight Price Classifier

[LIVE APP RUN](https://flightpriceclassify.streamlit.app/)

A Streamlit app that **tracks live flight prices** and classifies flights as **Cheaper or Costlier** using **Logistic Regression**.  
Built with the **Amadeus API** and `pandas`, `scikit-learn`, `seaborn`.

---

## 🚀 Features
- **Live flight search** using Amadeus API  
- **Price distribution visualizations**   
- **Logistic Regression classifier** for cheap vs costly  
- **Interactive dashboards** with Streamlit  

---

## 📦 Tech Stack
- **Frontend:** Streamlit, Matplotlib, Seaborn  
- **Backend:** Python, Amadeus API  
- **ML:** scikit-learn (Linear & Logistic Regression)  

---

## ⚡ Quickstart
```bash
# Clone
git clone https://github.com/<your-username>/FLIGHT.git
cd FLIGHT

# Create virtualenv (optional)
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Add your Amadeus credentials to .streamlit/secrets.toml
# Run
streamlit run flight_app.py
