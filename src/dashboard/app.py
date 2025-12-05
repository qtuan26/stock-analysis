import os
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Stock Dashboard", layout="wide")


# ROOT PROJECT
ROOT_DIR = Path(__file__).resolve().parents[2]

CLEAN_PATH = ROOT_DIR / "data" / "processed" / "clean"
FEATURE_PATH = ROOT_DIR / "data" / "processed" / "features"


#  KIỂM TRA THƯ MỤC
if not CLEAN_PATH.exists():
    st.error(f"❌ Không tìm thấy thư mục: {CLEAN_PATH}")
    st.stop()

# DANH SÁCH MÃ CỔ PHIẾU
available_stocks = sorted([f.stem for f in CLEAN_PATH.iterdir() if f.suffix == ".csv"])

if len(available_stocks) == 0:
    st.error("❌ Không có file CSV trong data/processed/clean/")
    st.stop()

# LOAD CLEAN DATA
@st.cache_data
def load_clean(symbol):
    df = None
    path = CLEAN_PATH / f"{symbol}.csv"

    if not path.exists():
        return None

    try:
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
    except:
        return None

    return df


# LOAD FEATURE DATA
@st.cache_data
def load_features(symbol):
    path = FEATURE_PATH / f"{symbol}_features.csv"

    if not path.exists():
        return None

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    return df


#  frontend
st.title("📊 Stock Analysis Dashboard")

stock = st.sidebar.selectbox("📈 Chọn mã cổ phiếu", available_stocks)

df = load_clean(stock)
df_features = load_features(stock)
if df is None or df.empty:
    st.error("❌ Không tìm thấy dữ liệu cổ phiếu")
    st.stop()
min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

start_date, end_date = st.sidebar.date_input(
    "📅 Chọn khoảng thời gian",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Lọc dữ liệu theo ngày
df = df[(df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)]

if df.empty:
    st.warning("⚠️ Không có dữ liệu trong khoảng ngày đã chọn.")
    st.stop()

latest = df.iloc[-1]
prev = df.iloc[-2]



#  TÍNH MA / RSI / MACD
# MA
df["MA20"] = df["Close"].rolling(window=20).mean()
df["MA50"] = df["Close"].rolling(window=50).mean()

# RSI
delta = df["Close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
df["RSI"] = 100 - (100 / (1 + rs))

# MACD
exp12 = df["Close"].ewm(span=12, adjust=False).mean()
exp26 = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = exp12 - exp26
df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

#KPI CARDS 
col1, col2, col3, col4 = st.columns(4)

price_diff = latest["Close"] - prev["Close"]
price_pct = price_diff / prev["Close"] * 100

col1.metric("Giá đóng cửa", f"${latest['Close']:.2f}", f"{price_diff:.2f} ({price_pct:.2f}%)")
col2.metric("Volume", f"{int(latest['Volume']):,}")
col3.metric("Cao nhất", f"${df['High'].max():.2f}")
col4.metric("Thấp nhất", f"${df['Low'].min():.2f}")


#  DOWNLOAD BÁO CÁO
st.download_button(
    label="⬇️ Tải dữ liệu CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name=f"{stock}_report.csv",
    mime="text/csv"
)

st.subheader("📈 Giá + Moving Average")

price_df = df.set_index("Date")[["Close", "MA20", "MA50"]]
st.line_chart(price_df)

st.subheader("📉 RSI")
st.line_chart(df.set_index("Date")[["RSI"]])

st.subheader("📉 MACD")
st.line_chart(df.set_index("Date")[["MACD", "Signal"]])

 
# TÍN HIỆU MUA / BÁN TỰ ĐỘNG
df["Trade_Signal"] = "HOLD"

buy_condition = (df["MACD"] > df["Signal"]) & (df["RSI"] < 30)
sell_condition = (df["MACD"] < df["Signal"]) & (df["RSI"] > 70)

df.loc[buy_condition, "Trade_Signal"] = "BUY"
df.loc[sell_condition, "Trade_Signal"] = "SELL"
st.subheader("📌 Tín hiệu giao dịch tự động")

latest_signal = df["Trade_Signal"].iloc[-1]
if latest_signal == "BUY":
    st.success("📢 Tín hiệu hiện tại: MUA")
elif latest_signal == "SELL":
    st.error("📢 Tín hiệu hiện tại: BÁN")
else:
    st.warning("📢 Tín hiệu hiện tại: GIỮ")

#  
# # BIỂU ĐỒ GIÁ
#  
# st.subheader("📈 Giá đóng cửa")
# st.line_chart(df.set_index("Date")["Close"])

# st.subheader("📊 Dữ liệu 5 dòng đầu")
# st.dataframe(df.head())

#  
# # ✅ 3. CHỈ BÁO KỸ THUẬT (FEATURES)
#  
# if df_features is not None and not df_features.empty:
#     st.subheader("📉 Các chỉ báo kỹ thuật")

#     df_plot = df_features.copy()
#     df_plot["Date"] = pd.to_datetime(df_plot["Date"])
#     df_plot = df_plot.set_index("Date")

#     df_plot = df_plot.select_dtypes(include=["float64", "int64"])

#     if not df_plot.empty:
#         st.line_chart(df_plot)
#     else:
#         st.info("ℹ️ File feature không có cột số.")
# else:
#     st.info("ℹ️ Chưa có dữ liệu chỉ số kỹ thuật.")

 
# THỐNG KÊ MÔ TẢ 
with st.expander("📊 Thống kê mô tả"):
    st.dataframe(df[["Open", "High", "Low", "Close", "Volume"]].describe())

