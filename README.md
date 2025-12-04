
1. PERFORMANCE METRICS QUA CÁC FOLDS 📈
pythonmetrics = ['acc', 'f1', 'auc']
Giải thích:

Walk-forward validation: Chia dữ liệu theo thời gian thành nhiều folds (mặc định 6 folds)
Mỗi fold train trên dữ liệu quá khứ, test trên dữ liệu tương lai
Vẽ 3 biểu đồ song song:

Accuracy: Tỷ lệ dự đoán đúng (up/down)
F1-Score: Trung bình điều hòa của Precision và Recall
AUC: Diện tích dưới ROC curve (khả năng phân biệt class)



Ý nghĩa:

Nếu metrics giảm dần qua các folds → Model overfitting hoặc thị trường thay đổi
Nếu metrics ổn định → Model robust với dữ liệu mới
So sánh LightGBM vs XGBoost → chọn model tốt hơn


2. ROC CURVES & PRECISION-RECALL CURVES 🎯
A. ROC Curve (Receiver Operating Characteristic)
pythonfpr, tpr, thresholds = roc_curve(y_true, y_proba_lgb)
Giải thích:

Trục X (FPR): False Positive Rate = dự đoán sai UP (thực tế DOWN)
Trục Y (TPR): True Positive Rate = dự đoán đúng UP
AUC = 0.5: Model random (đường chéo đen)
AUC > 0.7: Model khá tốt
AUC > 0.8: Model rất tốt

Ý nghĩa trading:

TPR cao → Bắt được nhiều tín hiệu UP đúng (profit opportunities)
FPR thấp → Ít tín hiệu sai → Ít loss

B. Precision-Recall Curve
pythonprecision, recall, _ = precision_recall_curve(y_true, y_proba_lgb)
Giải thích:

Precision: Trong số dự đoán UP, bao nhiêu % đúng?
Recall: Trong số thực tế UP, bắt được bao nhiêu %?
Average Precision (AP): Diện tích dưới PR curve

Ý nghĩa trading:

Precision cao → Ít False Signal → Ít bị loss khi vào lệnh
Recall cao → Không bỏ lỡ cơ hội profit
Quan trọng hơn ROC khi data imbalanced (số ngày tăng ≠ số ngày giảm)


3. CONFUSION MATRIX với NHIỀU THRESHOLD 🔲
pythonthresholds_to_plot = [0.3, 0.5, 0.7]
```

**Giải thích:**
Confusion Matrix cho mỗi threshold:
```
                Predicted DOWN    Predicted UP
Actual DOWN     TN (đúng)         FP (sai)
Actual UP       FN (bỏ lỡ)        TP (đúng)

Threshold = 0.3 (loose): Dự đoán UP nhiều → High Recall, Low Precision
Threshold = 0.5 (balanced): Cân bằng
Threshold = 0.7 (strict): Dự đoán UP ít → High Precision, Low Recall

Ý nghĩa trading:

Conservative trader (tránh risk): Chọn threshold cao (0.7) → Ít FP
Aggressive trader (không bỏ lỡ): Chọn threshold thấp (0.3) → Ít FN


4. FEATURE IMPORTANCE ANALYSIS 🎖️
A. Top 30 Features Bar Chart
pythontop_features = feature_importance.head(30)
Giải thích:

Importance (Gain): Mức độ giảm loss khi split tree theo feature này
Features có importance cao → Quyết định chính trong dự đoán
Ví dụ: RSI_14, MACD, SMA_20 thường quan trọng

B. Cumulative Importance
pythoncumsum_pct = cumsum / cumsum.iloc[-1] * 100
Giải thích:

Trục X: Số lượng features
Trục Y: % tổng importance tích lũy
80% threshold: Bao nhiêu features đóng góp 80% importance?
95% threshold: Bao nhiêu features đóng góp 95%?

Ý nghĩa:

Nếu 10 features đạt 80% → Có thể bỏ features ít quan trọng để tăng tốc
Principle: Pareto 80/20 trong feature selection

C. Feature Importance by Category
pythoncategories = ['Moving Averages', 'RSI', 'MACD', 'Bollinger Bands', ...]
```

**Giải thích:**
- Nhóm features theo loại indicator
- Xem nhóm nào quan trọng nhất

**Ví dụ kết quả:**
```
Moving Averages: 35%
MACD: 20%
RSI: 15%
Volume: 12%
...
→ Chiến lược nên tập trung vào Moving Averages

5. SHAP EXPLAINABILITY 🔍
A. SHAP Summary Plot (Beeswarm)
pythonshap.summary_plot(shap_values, X_sample_scaled, feature_names=feature_cols)
Giải thích từng thành phần:

Trục Y: Features được xếp theo importance (cao → thấp)
Trục X: SHAP value (impact lên prediction)

X > 0: Đẩy prediction lên (tăng xác suất UP)
X < 0: Kéo prediction xuống (tăng xác suất DOWN)


Màu sắc:

Đỏ: Feature value cao
Xanh: Feature value thấp



Đọc hiểu:

Ví dụ RSI_14:

Điểm đỏ ở X > 0 → RSI cao → Dự đoán UP
Điểm xanh ở X < 0 → RSI thấp → Dự đoán DOWN
Logic: RSI > 70 (overbought) có thể sắp DOWN, nhưng model học được xu hướng khác



B. SHAP Bar Plot
pythonshap.summary_plot(..., plot_type='bar')
Giải thích:

Chỉ hiển thị mean(|SHAP value|) = tầm ảnh hưởng trung bình
Khác với Feature Importance (dựa vào tree structure)
SHAP importance = impact thực tế lên từng prediction

So sánh:

Feature Importance: Quan trọng trong cấu trúc model
SHAP: Quan trọng trong từng dự đoán cụ thể


6. BACKTESTING VISUALIZATION 💰
A. Cumulative Returns
pythondaily['lgb_strat_ret_cum'] = (1 + daily['lgb_strat_ret']).cumprod()
Giải thích:

Strategy:

Dự đoán UP (signal=1) → Mua cổ phiếu
Dự đoán DOWN (signal=0) → Không giữ (hoặc short nếu cho phép)


Cumulative Return: Tổng lợi nhuận tích lũy từ đầu đến cuối
Công thức: (1 + r1) × (1 + r2) × ... × (1 + rn)

Đọc biểu đồ:

Đường đi lên → Strategy profitable
Đường nằm ngang/đi xuống → Strategy thua lỗ
So sánh LightGBM vs XGBoost → Chọn strategy tốt hơn

B. Drawdown Analysis
pythondrawdown = (cum_returns - running_max) / running_max
```

**Giải thích:**
- **Drawdown**: % sụt giảm từ đỉnh gần nhất
- **Max Drawdown (MDD)**: Sụt giảm lớn nhất trong lịch sử

**Ví dụ:**
```
Portfolio: $100 → $120 (đỉnh) → $90 (đáy)
Drawdown = ($90 - $120) / $120 = -25%
Ý nghĩa:

MDD = -10% → Strategy ổn định
MDD = -40% → Risk cao, khó tâm lý chịu đựng
Quan trọng hơn Total Return vì đo risk!

C. Rolling Sharpe Ratio
pythonsharpe = returns.rolling(60).mean() / returns.rolling(60).std() * sqrt(252)
Giải thích:

Sharpe Ratio: Return / Risk (càng cao càng tốt)
Annualized: Nhân với sqrt(252) để chuẩn hóa 1 năm
Rolling 60 days: Tính trên cửa sổ trượt 60 ngày

Thang đo:

Sharpe < 1: Kém
Sharpe 1-2: Tốt
Sharpe > 2: Rất tốt (hiếm)
Sharpe > 3: Xuất sắc (rất hiếm)

Đọc biểu đồ:

Sharpe biến động mạnh → Strategy không ổn định
Sharpe giảm dần → Strategy bị deteriorate theo thời gian


7. PREDICTION DISTRIBUTION ANALYSIS 📊
A. Distribution by Actual Class
pythonplt.hist(y_proba_lgb[y_true == 0], ...)  # Actual DOWN
plt.hist(y_proba_lgb[y_true == 1], ...)  # Actual UP
Giải thích:

Ideal: 2 histogram tách biệt rõ ràng

Actual DOWN → Predictions gần 0
Actual UP → Predictions gần 1


Poor model: 2 histogram overlap nhiều → Không phân biệt được

B. Calibration Plot
pythonprob_true.append(y_true[mask].mean())  # Actual frequency
prob_pred.append((lower + upper) / 2)   # Predicted probability
```

**Giải thích:**
- Chia predictions thành 10 bins (0-0.1, 0.1-0.2, ..., 0.9-1.0)
- Với mỗi bin: So sánh "predicted probability" vs "actual frequency"

**Ví dụ:**
```
Model dự đoán 0.7 (70% UP) cho 100 cases
→ Thực tế: 65 cases UP (65%)
→ Model hơi overconfident
Perfect calibration: Đường model trùng đường chéo
C. Average Prediction by Month
pythonmonthly_pred = df.groupby('month')['lgb_proba'].mean()
Giải thích:

Xem model có bias theo thời gian không?
Ví dụ: Luôn dự đoán UP trong tháng 1 (January effect)?

D. Prediction Confidence Distribution
pythonconfidence = np.abs(y_proba_lgb - 0.5) * 2
Giải thích:

Confidence = khoảng cách từ 0.5

Prediction = 0.1 hoặc 0.9 → Confidence = 0.8 (rất tự tin)
Prediction = 0.5 → Confidence = 0 (không chắc)


Phân bố lý tưởng: U-shape (nhiều predictions ở 2 đầu)


8. PER-SYMBOL PERFORMANCE 📈
pythonfor symbol in df['symbol'].unique():
    acc = accuracy_score(y_sym, y_pred_sym)
    auc = roc_auc_score(y_sym, y_pred_sym)
Giải thích:

Đánh giá model riêng cho từng cổ phiếu
Một số cổ phiếu dễ dự đoán hơn (higher AUC)
Một số cổ phiếu khó (lower AUC)

Ý nghĩa:

Best performers: Focus trading vào những cổ phiếu này
Worst performers: Tránh hoặc cần feature engineering riêng


🎯 TÓM TẮT MỤC ĐÍCH TỪNG PHẦN

![alt text](image.png)
```