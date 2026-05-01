import pandas as pd
import numpy as np
import glob
import os
import requests
from xgboost import XGBClassifier

# =========================
# CONFIG
# =========================
HOLD_DAYS = 3
TARGET_RETURN = 0.08
MIN_DATA = 120

# Buat folder output jika belum ada
os.makedirs("output", exist_ok=True)

# =========================
# TELEGRAM FUNCTION
# =========================
def send_telegram(message):
    # Mengambil kredensial dari environment variable GitHub
    token = os.getenv('TELEGRAM_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not token or not chat_id:
        print("Error: TELEGRAM_TOKEN atau TELEGRAM_CHAT_ID tidak ditemukan di environment!")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Notifikasi Telegram berhasil dikirim.")
        else:
            print(f"Gagal kirim Telegram: {response.text}")
    except Exception as e:
        print(f"Error saat mengirim Telegram: {e}")

# =========================
# LOAD + CLEAN
# =========================
def load_csv(file):
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]
    if 'Date' not in df.columns:
        df.rename(columns={'date': 'Date'}, inplace=True)
    df = df[['Date','Open','High','Low','Close','Volume']]
    for col in ['Open','High','Low','Close','Volume']:
        df[col] = df[col].astype(str).str.replace(',', '', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna().sort_values('Date')
    return df

# =========================
# FEATURE ENGINEERING
# =========================
def create_features(df):
    df = df.copy()
    df['ret_1'] = df['Close'].pct_change()
    df['ret_3'] = df['Close'].pct_change(3)
    df['ma20'] = df['Close'].rolling(20).mean()
    df['ma50'] = df['Close'].rolling(50).mean()
    df['trend'] = df['Close'] / df['ma50']
    df['ma_gap'] = (df['ma20'] - df['ma50']) / df['ma50']
    df['momentum_slope'] = df['Close'].rolling(5).mean().pct_change(5)
    df['vol_ma'] = df['Volume'].rolling(10).mean()
    df['vol_ratio'] = df['Volume'] / df['vol_ma']
    df['volatility'] = df['Close'].pct_change().rolling(10).std()
    df['range'] = (df['High'] - df['Low']) / df['Close']
    df['hh_10'] = df['High'].rolling(10).max()
    df['breakout'] = df['Close'] / df['hh_10']
    df['breakout_strength'] = df['Close'] - df['hh_10'].shift(1)
    
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'] / df['Close']
    return df

def create_label(df):
    future_close = df['Close'].shift(-HOLD_DAYS)
    df['target'] = (future_close / df['Close'] - 1) >= TARGET_RETURN
    return df

# =========================
# PROCESS DATA
# =========================
files = glob.glob("data/*.csv")
all_data = []

for file in files:
    try:
        df = load_csv(file)
        if len(df) < MIN_DATA: continue
        ticker = os.path.basename(file).split('_')[0]
        df['ticker'] = ticker
        df = create_features(df)
        df = create_label(df)
        all_data.append(df)
    except Exception as e:
        print(f"skip {file}: {e}")

if not all_data:
    send_telegram("❌ Screener Error: Tidak ada data CSV yang terbaca di folder data/")
    exit(0)

df_all = pd.concat(all_data, ignore_index=True).dropna()

features = ['ret_1','ret_3','trend','ma_gap','momentum_slope','vol_ratio','volatility','range','breakout','breakout_strength','rsi','atr_pct']
df_all = df_all.sort_values('Date')
split_date = df_all['Date'].quantile(0.7)
train_df = df_all[df_all['Date'] <= split_date]
test_df  = df_all[df_all['Date'] > split_date]

X_train = train_df[features]
y_train = train_df['target'].astype(int)

scale_pos_weight = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-9)
model = XGBClassifier(n_estimators=600, max_depth=4, learning_rate=0.03, scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss')
model.fit(X_train, y_train)

# =========================
# LIVE SCREENER
# =========================
latest_date = test_df['Date'].max()
df_live = test_df[test_df['Date'] == latest_date].copy()
df_live['prob'] = model.predict_proba(df_live[features])[:,1]

# Smart Filter
df_live = df_live[
    (df_live['prob'] > 0.65) & (df_live['trend'] > 1.02) & 
    (df_live['vol_ratio'] > 1.2) & (df_live['breakout'] > 0.98) & 
    (df_live['rsi'].between(55, 75))
]

# Scoring & Send Notif
if not df_live.empty:
    df_live['score'] = (df_live['prob'] * 0.5 + df_live['vol_ratio'] * 0.2 + df_live['momentum_slope'] * 0.2 + df_live['breakout'] * 0.1)
    df_live = df_live.sort_values(by='score', ascending=False).head(5)

    msg = f"🎯 *Screener Probability Hits!*\n📅 Tanggal: {latest_date.strftime('%Y-%m-%d')}\n\n"
    for _, row in df_live.iterrows():
        msg += f"• *{row['ticker']}*\n  Prob: {row['prob']:.2f} | Score: {row['score']:.2f}\n  RSI: {row['rsi']:.1f} | Vol: {row['vol_ratio']:.1f}\n\n"
    
    print(msg)
    send_telegram(msg)
    df_live.to_csv("output/live_signal.csv", index=False)
else:
    msg = f"😴 *Screener Selesai*\nTanggal: {latest_date.strftime('%Y-%m-%d')}\nTidak ada saham yang lolos kriteria hari ini."
    print(msg)
    send_telegram(msg)
