import pandas as pd

# Đọc CSV
df = pd.read_csv("data/raw/plotly_stocks.csv", parse_dates=["Date"])

# Xem tất cả ticker duy nhất
tickers = df["Stock"].unique()
print(tickers)
print(f"Tổng số ticker: {len(tickers)}")

# project/
# ├─ data/
# │   ├─ raw/                # chứa file csv từ fetch
# │   └─ processed/
# │       ├─ clean/
# │       └─ features/
# ├─ models/                 # chứa các model train_all.py đã tạo
# ├─ notebooks/              # Jupyter notebook trực quan
# ├─ src/
# │   ├─ data/               # code fetch, clean, feature, pipeline
# │   └─ ml/                 # các module ml (prepare, predict, train_all,...)
