from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import streamlit as st

st.set_page_config(page_title="Store Sales Forecaster")

BASE = Path(__file__).parent


@st.cache_resource
def load_model():
    return pickle.load(open(BASE / "sales_model.pkl", "rb"))


model = load_model()
features = list(model.feature_names_in_)
families = sorted(c[len("family_"):] for c in features if c.startswith("family_"))

st.title("🛒 Store Sales Forecaster")
st.write(
    "A gradient boosting model (trained on the Kaggle Store Sales dataset, Favorita grocery stores in Ecuador) "
    "forecasts daily unit sales of a product family in a store, from the calendar date and the number of items "
    "on promotion."
)

col1, col2 = st.columns(2)
with col1:
    store = st.number_input("Store number (1-54)", 1, 54, 1)
    family = st.selectbox("Product family", families, index=families.index("GROCERY I") if "GROCERY I" in families else 0)
with col2:
    start = st.date_input("Forecast start date", date(2017, 8, 16), min_value=date(2016, 1, 1), max_value=date(2018, 12, 31))
    promo = st.slider("Items on promotion (per day)", 0, 100, 0)

days = pd.date_range(start, periods=16)
rows = pd.DataFrame(0, index=range(len(days)), columns=features)
rows["store_nbr"] = store
rows["onpromotion"] = promo
rows["dow"] = days.dayofweek
rows["day"] = days.day
rows["month"] = days.month
rows["year"] = days.year
rows[f"family_{family}"] = 1

pred = np.expm1(np.clip(model.predict(rows[features]), 0, None))
result = pd.DataFrame({"date": days.strftime("%a %d %b %Y"), "predicted sales": pred.round(1)})

st.metric(f"Predicted sales on {days[0].strftime('%d %b %Y')}", f"{pred[0]:,.0f} units")
st.line_chart(pd.Series(pred, index=days, name="Predicted daily sales"))
st.dataframe(result, hide_index=True, width="stretch")

st.caption(
    "Model: HistGradientBoosting on log-transformed sales (validation RMSLE ≈ 0.52 on the last 16 days of the Kaggle "
    "training data). The model only knows the calendar, the store, the product family, and promotions, so it cannot "
    "react to holidays, oil prices, or recent trends. Dates far outside 2016-2017 are extrapolations."
)
