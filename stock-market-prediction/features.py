import pandas as pd
import numpy as np
import ta  # thư viện technical analysis
from sklearn.preprocessing import StandardScaler

# Load dữ liệu gốc
df = pd.read_csv('price_train.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['symbol', 'date'])
def add_moving_averages(df):
    """Tính các đường trung bình động"""
    for symbol in df['symbol'].unique():
        mask = df['symbol'] == symbol
        
        # Simple Moving Averages
        df.loc[mask, 'SMA_5'] = df.loc[mask, 'close'].rolling(5).mean()
        df.loc[mask, 'SMA_10'] = df.loc[mask, 'close'].rolling(10).mean()
        df.loc[mask, 'SMA_20'] = df.loc[mask, 'close'].rolling(20).mean()
        df.loc[mask, 'SMA_50'] = df.loc[mask, 'close'].rolling(50).mean()
        
        # Exponential Moving Averages
        df.loc[mask, 'EMA_12'] = df.loc[mask, 'close'].ewm(span=12).mean()
        df.loc[mask, 'EMA_26'] = df.loc[mask, 'close'].ewm(span=26).mean()
    
    return df
def add_rsi(df, periods=[14, 21]):
    """Tính chỉ số RSI"""
    for symbol in df['symbol'].unique():
        mask = df['symbol'] == symbol
        for period in periods:
            df.loc[mask, f'RSI_{period}'] = ta.momentum.RSIIndicator(
                df.loc[mask, 'close'], 
                window=period
            ).rsi()
    return df
def add_macd(df):
    """Tính MACD"""
    for symbol in df['symbol'].unique():
        mask = df['symbol'] == symbol
        
        macd = ta.trend.MACD(df.loc[mask, 'close'])
        df.loc[mask, 'MACD'] = macd.macd()
        df.loc[mask, 'MACD_signal'] = macd.macd_signal()
        df.loc[mask, 'MACD_diff'] = macd.macd_diff()
    
    return df
def add_bollinger_bands(df, window=20):
    """Tính Bollinger Bands"""
    for symbol in df['symbol'].unique():
        mask = df['symbol'] == symbol
        
        bollinger = ta.volatility.BollingerBands(
            df.loc[mask, 'close'], 
            window=window
        )
        df.loc[mask, 'BB_high'] = bollinger.bollinger_hband()
        df.loc[mask, 'BB_mid'] = bollinger.bollinger_mavg()
        df.loc[mask, 'BB_low'] = bollinger.bollinger_lband()
        df.loc[mask, 'BB_width'] = bollinger.bollinger_wband()
    
    return df
def add_stochastic(df):
    """Tính Stochastic Oscillator"""
    for symbol in df['symbol'].unique():
        mask = df['symbol'] == symbol
        
        stoch = ta.momentum.StochasticOscillator(
            df.loc[mask, 'high'],
            df.loc[mask, 'low'],
            df.loc[mask, 'close']
        )
        df.loc[mask, 'STOCH_K'] = stoch.stoch()
        df.loc[mask, 'STOCH_D'] = stoch.stoch_signal()
    
    return df
def add_adx(df):
    """Tính ADX - đo lường xu hướng"""
    for symbol in df['symbol'].unique():
        mask = df['symbol'] == symbol
        
        adx = ta.trend.ADXIndicator(
            df.loc[mask, 'high'],
            df.loc[mask, 'low'],
            df.loc[mask, 'close']
        )
        df.loc[mask, 'ADX'] = adx.adx()
    
    return df
def add_advanced_features(df):
    """Tạo features phức tạp hơn"""
    for symbol in df['symbol'].unique():
        mask = df['symbol'] == symbol
        
        # 1. Price-based features
        df.loc[mask, 'price_range'] = df.loc[mask, 'high'] - df.loc[mask, 'low']
        df.loc[mask, 'price_change'] = df.loc[mask, 'close'] - df.loc[mask, 'open']
        df.loc[mask, 'high_low_ratio'] = df.loc[mask, 'high'] / df.loc[mask, 'low']
        
        # 2. Returns
        df.loc[mask, 'daily_return'] = df.loc[mask, 'close'].pct_change()
        df.loc[mask, 'log_return'] = np.log(df.loc[mask, 'close'] / df.loc[mask, 'close'].shift(1))
        
        # 3. Volatility
        df.loc[mask, 'volatility_10'] = df.loc[mask, 'daily_return'].rolling(10).std()
        df.loc[mask, 'volatility_30'] = df.loc[mask, 'daily_return'].rolling(30).std()
        
        # 4. Volume features
        df.loc[mask, 'volume_ma_10'] = df.loc[mask, 'volume'].rolling(10).mean()
        df.loc[mask, 'volume_ratio'] = df.loc[mask, 'volume'] / df.loc[mask, 'volume_ma_10']
        
        # 5. Momentum features
        df.loc[mask, 'momentum_5'] = df.loc[mask, 'close'] - df.loc[mask, 'close'].shift(5)
        df.loc[mask, 'momentum_10'] = df.loc[mask, 'close'] - df.loc[mask, 'close'].shift(10)
        
        # 6. Trend features
        df.loc[mask, 'MA_cross'] = (df.loc[mask, 'SMA_10'] > df.loc[mask, 'SMA_20']).astype(int)
        
    return df
def create_labels(df, forecast_horizon=1):
    """Tạo labels cho bài toán phân loại và regression"""
    for symbol in df['symbol'].unique():
        mask = df['symbol'] == symbol
        
        # Classification: Giá tăng (1) hay giảm (0)
        df.loc[mask, 'label_up_down'] = (
            df.loc[mask, 'close'].shift(-forecast_horizon) > df.loc[mask, 'close']
        ).astype(int)
        
        # Regression: % thay đổi giá
        df.loc[mask, 'target_return'] = (
            (df.loc[mask, 'close'].shift(-forecast_horizon) - df.loc[mask, 'close']) 
            / df.loc[mask, 'close'] * 100
        )
    
    return df
def create_features_pipeline(input_file='price_train.csv', 
                             output_file='features_engineered.csv'):
    """Pipeline tạo features hoàn chỉnh"""
    
    print("📥 Loading data...")
    df = pd.read_csv(input_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date'])
    
    print("📊 Adding Moving Averages...")
    df = add_moving_averages(df)
    
    print("📈 Adding RSI...")
    df = add_rsi(df)
    
    print("📉 Adding MACD...")
    df = add_macd(df)
    
    print("🎯 Adding Bollinger Bands...")
    df = add_bollinger_bands(df)
    
    print("⚡ Adding Stochastic...")
    df = add_stochastic(df)
    
    print("🎲 Adding ADX...")
    df = add_adx(df)
    
    print("🔧 Adding Advanced Features...")
    df = add_advanced_features(df)
    
    print("🏷️ Creating Labels...")
    df = create_labels(df)
    
    # Loại bỏ NaN
    print("🧹 Cleaning data...")
    df = df.dropna()
    
    # Lưu file
    print(f"💾 Saving to {output_file}...")
    df.to_csv(output_file, index=False)
    
    print(f"✅ Done! Created {len(df.columns)} features for {df['symbol'].nunique()} symbols")
    print(f"📊 Features: {df.columns.tolist()}")
    
    return df

# Chạy pipeline
df_features = create_features_pipeline()