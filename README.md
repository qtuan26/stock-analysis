```
Cấu trúc dự án
bai-cuoi-ky/
├── src/
│   └── data/
│       ├── fetch.py       # Download dữ liệu raw từ Yahoo Finance
│       ├── clean.py       # Làm sạch dữ liệu raw
│       ├── features.py    # Tính các chỉ báo kỹ thuật
│       └── pipeline.py    # Chạy toàn bộ pipeline
├── data/
│   ├── raw/               # CSV dữ liệu gốc
│   └── processed/
│       ├── clean/         # CSV dữ liệu đã clean
│       └── features/      # CSV có các chỉ báo kỹ thuật
├── requirements.txt
└── README.md
```

- Hướng dẫn cài đặt

Clone repo: <br>

git clone <repo-url> <br>
cd abc... <br>


- Tạo môi trường ảo:

python -m venv venv <br>

venv\Scripts\activate <br>

- Cài thư viện: 
pip install -r requirements.txt <br>

- Cách chạy pipeline
python src/data/pipeline.py <br>

```
Pipeline thực hiện tuần tự 3 bước:
Fetch
Lấy dữ liệu cổ phiếu từ Yahoo Finance.
Xử lý MultiIndex columns, thêm cột Ticker.
Lưu CSV raw vào data/raw/.
Clean
Chuẩn hóa cột Date.
Chọn các cột quan trọng: Open, High, Low, Close, Adj Close, Volume, Ticker.
Drop các dòng thiếu dữ liệu ở cột quan trọng.
Sort theo Ticker + Date.
Lưu CSV clean vào data/processed/clean/.
Features
Tính các chỉ báo kỹ thuật: RSI(14), SMA(20), EMA(12), Bollinger Bands(20).
Drop các dòng đầu bị NaN (do các chỉ báo cần “warm-up”).
Lưu CSV features vào data/processed/features/.
⚠️ Lưu ý: File features.csv sẽ mất khoảng 20 dòng đầu do tính toán các chỉ báo kỹ thuật, đây là hành vi bình thường.
🔹 Thêm ticker mới
Mở src/pipeline.py và sửa danh sách TICKERS:
TICKERS = ["AAPL", "AMZN", "GOOG", "MSFT", "TSLA", "NVDA"]
Chạy lại pipeline.
```