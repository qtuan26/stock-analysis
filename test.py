import pandas as pd

# Đọc CSV
df = pd.read_csv("data/raw/plotly_stocks.csv", parse_dates=["Date"])

# Xem tất cả ticker duy nhất
tickers = df["Stock"].unique()
print(tickers)
print(f"Tổng số ticker: {len(tickers)}")