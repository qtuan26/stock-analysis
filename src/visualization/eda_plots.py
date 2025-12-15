import matplotlib.pyplot as plt

def plot_price(df):
    fig, ax = plt.subplots()
    ax.plot(df["Date"], df["Close"])
    ax.set_title("Giá đóng cửa")
    ax.set_xlabel("Ngày")
    ax.set_ylabel("Giá")
    return fig


def plot_volume(df):
    fig, ax = plt.subplots()
    ax.bar(df["Date"], df["Volume"])
    ax.set_title("Khối lượng giao dịch")
    ax.set_xlabel("Ngày")
    ax.set_ylabel("Volume")
    return fig


def plot_return(df):
    df = df.copy()

    
    if "Daily_Return" not in df.columns:
        df["Daily_Return"] = df["Close"].pct_change()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["Daily_Return"].dropna(), bins=50)
    ax.set_title("Phân phối tỷ suất sinh lời")
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Frequency")

    return fig



def plot_ma(df):
    fig, ax = plt.subplots()
    ax.plot(df["Date"], df["Close"], label="Close")

    if "MA20" in df.columns:
        ax.plot(df["Date"], df["MA20"], label="MA20")
    if "MA50" in df.columns:
        ax.plot(df["Date"], df["MA50"], label="MA50")

    ax.legend()
    ax.set_title("Moving Average")
    return fig
