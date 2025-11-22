GIỚI THIỆU:
- Dự án này được thực hiện trong khuôn khổ môn học Python, với mục tiêu xây dựng một ứng dụng phân tích dữ liệu chứng khoán hoàn chỉnh, từ khâu thu thập dữ liệu, phân tích, xây dựng mô hình dự đoán đến hiển thị dashboard.
- Dự án gồm 5 thành viên, mỗi người phụ trách một phần riêng biệt tương ứng với workflow của một quy trình Data Science.

MỤC TIÊU CHÍNH
- Thu thập dữ liệu chứng khoán từ API hoặc CSV.
- Làm sạch, chuẩn hoá và lưu dữ liệu dưới dạng CSV.
- Thực hiện EDA (phân tích dữ liệu khám phá).
- Tính toán các chỉ báo kỹ thuật (MA, RSI, MACD…).
- Xây dựng mô hình Machine Learning đơn giản dự đoán giá.
- Tạo dashboard bằng Streamlit để trình diễn kết quả.


HƯỚNG DẪN CÀI ĐẶT THƯ VIỆN TRONG requirements.txt:
- Bước 1 — Kiểm tra phiên bản Python
Dự án này yêu cầu Python ≥ 3.9.
- Kiểm tra bằng lệnh:
python --version

- Bước 2 - Tạo môi trường ảo (venv)
python -m venv venv
- Kích hoạt:
venv\Scripts\activate

- Bước 3 - Cài thư viện từ requirements.txt
-Trong trạng thái đang bật venv, chạy:
pip install -r requirements.txt
- Tuyệt đối không chạy pip install bên ngoài venv.

- Bước 4 — Chạy code hoặc Jupyter Notebook:
...

```text
CẤU TRÚC PROJECT:
stock-analysis-project/
│
├── venv            # môi trường ảo
|
├── data/
│   ├── raw/                # dữ liệu thô từ API (Người 1)
│   ├── processed/          # CSV sau cleaning (Người 1)
│   └── external/           # dữ liệu tham khảo (tin tức, index,...)
│
├── notebooks/
│   ├── 01_data_collection.ipynb     # Người 1
│   ├── 02_eda_visualization.ipynb   # Người 2
│   ├── 03_features_indicators.ipynb # Người 3
│   ├── 04_ml_models.ipynb           # Người 4
│   └── 05_dashboard_demo.ipynb      # Người 5
│
├── src/
│   ├── data/               # Script Python xử lý dữ liệu
│   │   ├── fetch_data.py
│   │   ├── clean_data.py
│   │   └── pipeline.py
│   │
│   ├── eda/                # Code trực quan hóa (Người 2)
│   │   ├── plots.py
│   │   └── statistics.py
│   │
│   ├── features/           # Indicators + feature engineering (Người 3)
│   │   ├── indicators.py
│   │   └── feature_engineering.py
│   │
│   ├── models/             # Machine Learning (Người 4)
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   │
│   └── dashboard/          # Streamlit app (Người 5)
│       ├── app.py
│       └── components/
│
│
├── requirements.txt         # thư viện cần cài
├── README.md                # mô tả project
└── .gitignore               # bỏ qua data thô, file nặng
```