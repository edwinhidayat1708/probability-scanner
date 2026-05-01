import pandas as pd
import numpy as np
import glob
import os
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
# LOAD + CLEAN
# =========================
def load_csv(file):
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]

    if 'Date' not in df.columns:
        df.rename(columns={'date': 'Date'}, inplace=True)

    df = df[['Date','Open','High','Low','Close','Volume']]

    for col in ['Open','High','Low','Close','Volume']:
        df[col] = (
            df[col].astype(str)
            .str.replace(',', '', regex=False)
        )
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

# =========================
# LABEL
# =========================
def create_label(df):
    future_close = df['Close'].shift(-HOLD_DAYS)
    df['target'] = (future_close / df['Close'] - 1) >= TARGET_RETURN
    return df

# =========================
# LOAD ALL
# =========================
files = glob.glob("data/*.csv")
all_data = []

print(f"Ditemukan {len(files)} file CSV.")

for file in files:
    try:
        df = load_csv(file)
        if len(df) < MIN_DATA:
            continue
        
        ticker = os.path.basename(file).split('_')[0]
        df['ticker'] = ticker
        df = create_features(df)
        df = create_label(df)
        all_data.append(df)
    except Exception as e:
        print(f"skip {file}: {e}")

# FIX: Cek jika data kosong agar tidak error ValueError
if not all_data:
    print("Error: Tidak ada data saham yang berhasil dimuat. Pastikan folder 'data/' berisi file CSV.")
    exit(0)

df_all = pd.concat(all_data, ignore_index=True)
df_all = df_all.dropna()

if df_all.empty:
    print("Error: Dataframe kosong setelah dropna().")
    exit(0)

# =========================
# FEATURES
# =========================
features = [
    'ret_1','ret_3', 'trend','ma_gap', 'momentum_slope',
    'vol_ratio', 'volatility','range', 'breakout',
    'breakout_strength', 'rsi', 'atr_pct'
]

# =========================
# SPLIT (TIME SAFE)
# =========================
df_all = df_all.sort_values('Date')
split_date = df_all['Date'].quantile(0.7)

train_df = df_all[df_all['Date'] <= split_date]
test_df  = df_all[df_all['Date'] > split_date]

if train_df.empty or test_df.empty:
    print("Error: Data tidak cukup untuk split training/testing.")
    exit(0)

X_train = train_df[features]
y_train = train_df['target'].astype(int)
X_test = test_df[features]

# =========================
# MODEL
# =========================
scale_pos_weight = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-9)

model = XGBClassifier(
    n_estimators=600,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1,
    reg_lambda=2,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)

# =========================
# LIVE SCREENER
# =========================
latest_date = test_df['Date'].max()
df_live = test_df[test_df['Date'] == latest_date].copy()

if df_live.empty:
    print("Error: Tidak ada data untuk tanggal terbaru.")
    exit(0)

df_live['prob'] = model.predict_proba(df_live[features])[:,1]

# =========================
# SMART FILTER
# =========================
df_live = df_live[
    (df_live['prob'] > 0.65) &
    (df_live['trend'] > 1.02) &
    (df_live['vol_ratio'] > 1.2) &
    (df_live['breakout'] > 0.98) &
    (df_live['rsi'].between(55, 75))
]

# =========================
# SCORING & RANKING
# =========================
if not df_live.empty:
    df_live['score'] = (
        df_live['prob'] * 0.5 +
        df_live['vol_ratio'] * 0.2 +
        df_live['momentum_slope'] * 0.2 +
        df_live['breakout'] * 0.1
    )
    df_live = df_live.sort_values(by='score', ascending=False)
    df_live = df_live.groupby('ticker').head(1)
    df_live = df_live.head(5)

    print("\n=== LIVE SCREENER (IMPROVED) ===")
    print(df_live[['ticker','prob','score','rsi','vol_ratio','breakout']])
    df_live.to_csv("output/live_signal.csv", index=False)
else:
    print("\n=== LIVE SCREENER ===")
    print("Tidak ada saham yang lolos filter hari ini.")
