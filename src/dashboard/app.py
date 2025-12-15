import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

# =========================
# FIX ROOT PROJECT
# =========================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_PATH = os.path.join(ROOT_DIR, "stock-market-prediction","eda_trucquanhoa", "price_train.csv")

# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(layout="wide")
st.title("📊 Phân tích cổ phiếu Việt Nam (HOSE)")

# =========================
# LOAD DATA
# =========================
if not os.path.exists(DATA_PATH):
    st.error(f"❌ Không tìm thấy file: {DATA_PATH}")
    st.stop()

df = pd.read_csv(DATA_PATH)

# Chuẩn hóa tên cột theo chương 4
df.columns = df.columns.str.lower()
df["date"] = pd.to_datetime(df["date"])

# =========================
# CHỌN MÃ CỔ PHIẾU
# =========================
symbols = sorted(df["symbol"].unique())
stock = st.sidebar.selectbox("📈 Chọn mã cổ phiếu", symbols)

df_stock = df[df["symbol"] == stock].copy()
df_stock = df_stock.sort_values("date")

# =========================
# TÍNH TOÁN CHỈ BÁO
# =========================
df_stock["daily_return"] = df_stock["close"].pct_change()
df_stock["volatility"] = df_stock["daily_return"].rolling(20).std()
df_stock["cum_return"] = (1 + df_stock["daily_return"]).cumprod()

# =========================
# MENU
# =========================
menu = st.sidebar.radio(
    "Chọn loại phân tích",
    [
        "Giá đóng cửa",
        "Khối lượng",
        "Candlestick",
        "Tỷ suất sinh lời (Daily Return)",
        "Độ biến động (Volatility)",
        "Lợi nhuận lũy kế"
    ]
)

# =========================
# BIỂU ĐỒ
# =========================
if menu == "Giá đóng cửa":
    st.subheader(f"📈 Giá đóng cửa – {stock}")
    fig, ax = plt.subplots()
    ax.plot(df_stock["date"], df_stock["close"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Giá")
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

elif menu == "Khối lượng":
    st.subheader(f"📊 Khối lượng giao dịch – {stock}")
    fig, ax = plt.subplots()
    ax.bar(df_stock["date"], df_stock["volume"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Volume")
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

elif menu == "Candlestick":
    st.subheader(f"🕯️ Biểu đồ nến – {stock}")
    fig, ax = plt.subplots()

    up = df_stock[df_stock["close"] >= df_stock["open"]]
    down = df_stock[df_stock["close"] < df_stock["open"]]

    ax.bar(up["date"], up["close"] - up["open"], bottom=up["open"])
    ax.bar(up["date"], up["high"] - up["close"], bottom=up["close"])
    ax.bar(up["date"], up["open"] - up["low"], bottom=up["low"])

    ax.bar(down["date"], down["close"] - down["open"], bottom=down["open"])
    ax.bar(down["date"], down["high"] - down["open"], bottom=down["open"])
    ax.bar(down["date"], down["close"] - down["low"], bottom=down["low"])

    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

elif menu == "Tỷ suất sinh lời (Daily Return)":
    st.subheader(f"📉 Phân phối Tỷ suất sinh lời – {stock}")
    fig, ax = plt.subplots()
    ax.hist(df_stock["daily_return"].dropna(), bins=50)
    
    st.pyplot(fig)

elif menu == "Độ biến động (Volatility)":
    st.subheader(f"⚡ Độ biến động – {stock}")
    fig, ax = plt.subplots()
    ax.plot(df_stock["date"], df_stock["volatility"])
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

elif menu == "Lợi nhuận lũy kế":
    st.subheader(f"📈 Lợi nhuận lũy kế – {stock}")
    fig, ax = plt.subplots()
    ax.plot(df_stock["date"], df_stock["cum_return"])
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)



st.subheader("⬇️ Tải dữ liệu cổ phiếu")

csv_data = df_stock.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Tải dữ liệu CSV",
    data=csv_data,
    file_name=f"{stock}_data.csv",
    mime="text/csv"
)
