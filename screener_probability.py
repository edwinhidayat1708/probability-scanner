import pandas as pd
import numpy as np
import os
import requests
import yfinance as yf
from xgboost import XGBClassifier
from datetime import datetime

# =========================
# CONFIG
# =========================
HOLD_DAYS = 3
TARGET_RETURN = 0.08
MIN_DATA = 120

# Daftar saham pilihan (Papan Utama & Pengembangan yang Likuid)
# Anda bisa menambah daftar ini sesuai kebutuhan
TICKERS_IDX = [
    'ADRO.JK', 'AMRT.JK', 'ANKM.JK', 'ASII.JK', 'BBCA.JK', 'BBNI.JK', 'BBRI.JK', 
    'BMRI.JK', 'BRPT.JK', 'CPIN.JK', 'GOTO.JK', 'INKP.JK', 'ITMG.JK', 'KLBF.JK', 
    'MDKA.JK', 'PGAS.JK', 'PTBA.JK', 'TLKM.JK', 'TPIA.JK', 'UNTR.JK', 'UNVR.JK',
    'ADMR.JK', 'AKRA.JK', 'ANTM.JK', 'ARTO.JK', 'BRIS.JK', 'BUKA.JK', 'CPIN.JK',
    'ESSA.JK', 'HRUM.JK', 'ICBP.JK', 'INCO.JK', 'INDF.JK', 'MEDC.JK', 'MIKA.JK',
    ' PGEO.JK', 'PTMP.JK', 'SMGR.JK', 'TOWR.JK', 'MBMA.JK', 'ACES.JK', 'BSDE.JK',
    'PWON.JK', 'CTRA.JK', 'ELSA.JK', 'MYOR.JK', 'JSMR.JK', 'EXCL.JK'
]

os.makedirs("output", exist_ok=True)

# =========================
# TELEGRAM FUNCTION
# =========================
def send_telegram(message):
    token = os.getenv('TELEGRAM_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    if not token or not chat_id:
        print("Kredensial Telegram tidak ditemukan.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Error kirim Telegram: {e}")

# =========================
# FEATURE ENGINEERING
# =========================
def create_features(df):
    df = df.copy()
    # Mengambil level kolom jika multi-index (sering terjadi di yfinance bulk)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df['ret_1'] = df['Close'].pct_change()
    df['ret_3'] = df['Close'].pct_change(3)
    df['ma20'] = df['Close'].rolling(20).mean()
    df['ma50'] = df['Close'].rolling(50).mean()
    df['trend'] = df['Close'] / df['ma50']
    df['ma_gap'] = (df['ma20'] - df['ma50']) / df['ma50']
    df['momentum_slope'] = df['Close'].rolling(5).mean().pct_change(5)
    df['vol_ma'] = df['Volume'].rolling(10).mean()
    df['vol_ratio'] = df['Volume'] / (df['vol_ma'] + 1e-9)
    df['volatility'] = df['Close'].pct_change().rolling(10).std()
    df['range'] = (df['High'] - df['Low']) / (df['Close'] + 1e-9)
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
    df['atr_pct'] = df['atr'] / (df['Close'] + 1e-9)
    return df

def create_label(df):
    future_close = df['Close'].shift(-HOLD_DAYS)
    df['target'] = (future_close / df['Close'] - 1) >= TARGET_RETURN
    return df

# =========================
# DOWNLOAD & PROCESS
# =========================
print(f"Downloading data for {len(TICKERS_IDX)} tickers...")
# Download massal 1 tahun data
raw_data = yf.download(TICKERS_IDX, period="1y", interval="1d", group_by='ticker', progress=False)

all_data = []
for ticker in TICKERS_IDX:
    try:
        # Ambil data per ticker dari hasil bulk download
        if len(TICKERS_IDX) > 1:
            df = raw_data[ticker].copy()
        else:
            df = raw_data.copy()

        df = df.dropna(subset=['Close'])
        if len(df) < MIN_DATA:
            continue
            
        df = df.reset_index()
        df['ticker'] = ticker.replace('.JK', '')
        df = create_features(df)
        df = create_label(df)
        all_data.append(df)
    except Exception as e:
        print(f"Error processing {ticker}: {e}")

if not all_data:
    send_telegram("❌ Screener Error: Tidak ada data yang berhasil di-download.")
    exit(0)

df_all = pd.concat(all_data, ignore_index=True).dropna(subset=['rsi', 'trend'])

# =========================
# ML MODEL & SCREENING
# =========================
features = ['ret_1','ret_3','trend','ma_gap','momentum_slope','vol_ratio','volatility','range','breakout','breakout_strength','rsi','atr_pct']
df_all = df_all.sort_values('Date')
split_date = df_all['Date'].quantile(0.7)
train_df = df_all[df_all['Date'] <= split_date]
test_df  = df_all[df_all['Date'] > split_date]

X_train = train_df[features]
y_train = train_df['target'].astype(int)

scale_pos_weight = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-9)
model = XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss')
model.fit(X_train, y_train)

# Live Screener pada Tanggal Terakhir
latest_date = df_all['Date'].max()
df_live = df_all[df_all['Date'] == latest_date].copy()
df_live['prob'] = model.predict_proba(df_live[features])[:,1]

# Filter Strategi
df_final = df_live[
    (df_live['prob'] > 0.60) & 
    (df_live['trend'] > 1.0) & 
    (df_live['vol_ratio'] > 1.1) &
    (df_live['rsi'].between(45, 75))
].copy()

# Notifikasi
if not df_final.empty:
    df_final['score'] = (df_final['prob'] * 0.5 + df_final['vol_ratio'] * 0.3 + df_final['breakout'] * 0.2)
    df_final = df_final.sort_values(by='score', ascending=False).head(5)

    msg = f"🎯 *IDX PROBABILITY HITS*\n📅 {latest_date.strftime('%Y-%m-%d')}\n\n"
    for _, row in df_final.iterrows():
        msg += f"• *{row['ticker']}*\n  Prob: {row['prob']:.2f} | Score: {row['score']:.2f}\n  RSI: {row['rsi']:.1f} | Vol: {row['vol_ratio']:.1f}\n\n"
    send_telegram(msg)
    df_final.to_csv("output/live_signal.csv", index=False)
else:
    send_telegram(f"😴 *Screener Selesai*\n{latest_date.strftime('%Y-%m-%d')}\nTidak ada saham yang masuk kriteria.")
