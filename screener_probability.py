import pandas as pd
import numpy as np
import os
import requests
import yfinance as yf
import time
from xgboost import XGBClassifier
from datetime import datetime

# =========================
# CONFIG (Logic Intact)
# =========================
HOLD_DAYS = 3
TARGET_RETURN = 0.08
MIN_DATA = 120
CHUNK_SIZE = 25  # Download per 25 emiten agar aman

# LIST TICKER (Sudah difilter untuk yang relatif aktif)
TICKERS_IDX = [
    'AALI.JK', 'ADRO.JK', 'ANTM.JK', 'ASII.JK', 'BBCA.JK', 'BBNI.JK', 'BBRI.JK', 'BBTN.JK', 
    'BMRI.JK', 'BRPT.JK', 'BSDE.JK', 'CPIN.JK', 'GOTO.JK', 'HRUM.JK', 'ICBP.JK', 'INCO.JK', 
    'INDF.JK', 'INKP.JK', 'ITMG.JK', 'KLBF.JK', 'MDKA.JK', 'MEDC.JK', 'PGAS.JK', 'PTBA.JK', 
    'SMGR.JK', 'TLKM.JK', 'TPIA.JK', 'UNTR.JK', 'UNVR.JK', 'WIKA.JK', 'ACES.JK', 'AKRA.JK',
    'AMRT.JK', 'ANJT.JK', 'ARTO.JK', 'ASSA.JK', 'AUTO.JK', 'AVIA.JK', 'BFIN.JK', 'BRMS.JK',
    'BUKA.JK', 'CTRA.JK', 'ESSA.JK', 'EXCL.JK', 'HEAL.JK', 'JPFA.JK', 'JSMR.JK', 'LSIP.JK',
    'MAPI.JK', 'MYOR.JK', 'PGEO.JK', 'PTPP.JK', 'PWON.JK', 'RAJA.JK', 'SIDO.JK', 'SMRA.JK',
    'TKIM.JK', 'TOWR.JK'
    # ... Tambahkan ticker Papan Utama lainnya secara bertahap jika ini berhasil
]

def send_telegram(message):
    token = os.getenv('TELEGRAM_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    if not token or not chat_id: return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try: requests.post(url, json=payload, timeout=10)
    except: pass

def create_features(df):
    df = df.copy()
    # Handle Multi-index yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.ffill()
    # FEATURE CORE (Jangan dirubah)
    df['ret_1'] = df['Close'].pct_change()
    df['ret_3'] = df['Close'].pct_change(3)
    df['ma20'] = df['Close'].rolling(20).mean()
    df['ma50'] = df['Close'].rolling(50).mean()
    df['trend'] = df['Close'] / (df['ma50'] + 1e-9)
    df['ma_ratio'] = df['Close'] / (df['ma20'] + 1e-9)
    df['vol_ma'] = df['Volume'].rolling(10).mean()
    df['vol_ratio'] = df['Volume'] / (df['vol_ma'] + 1e-9)
    df['volatility'] = df['Close'].pct_change().rolling(10).std()
    df['range'] = (df['High'] - df['Low']) / (df['Close'] + 1e-9)
    df['hh_10'] = df['High'].rolling(10).max()
    df['breakout'] = df['Close'] / (df['hh_10'] + 1e-9)
    df['breakout_strength'] = df['Close'] - df['hh_10'].shift(1)
    
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

def create_label(df):
    future_close = df['Close'].shift(-HOLD_DAYS)
    df['target'] = (future_close / df['Close'] - 1) >= TARGET_RETURN
    return df

# ==========================================
# SAFE DOWNLOAD PROCESS (The Fix)
# ==========================================
print("Mendownload data IHSG...")
ihsg_data = yf.download('^JKSE', period="1y", interval="1d", progress=False)
ihsg_context = pd.DataFrame(index=ihsg_data.index)
ihsg_context['ihsg_ret_3'] = ihsg_data['Close'].pct_change(3)
ihsg_context['ihsg_trend'] = ihsg_data['Close'] / ihsg_data['Close'].rolling(50).mean()
ihsg_context = ihsg_context.ffill()

all_data = []
# Bagi ticker menjadi potongan kecil (Chunks)
for i in range(0, len(TICKERS_IDX), CHUNK_SIZE):
    chunk = TICKERS_IDX[i:i + CHUNK_SIZE]
    print(f"Mendownload chunk {i//CHUNK_SIZE + 1}...")
    
    try:
        # Download per chunk
        data_chunk = yf.download(chunk, period="1y", interval="1d", group_by='ticker', progress=False)
        
        for ticker in chunk:
            try:
                if len(chunk) > 1:
                    df = data_chunk[ticker].copy()
                else:
                    df = data_chunk.copy()
                
                df = df.dropna(subset=['Close'])
                # Filter Likuiditas Dasar: Volume tidak boleh 0 dalam 5 hari terakhir
                if len(df) < MIN_DATA or df['Volume'].tail(5).mean() < 100: 
                    continue
                
                df = df.reset_index()
                df['ticker'] = ticker.replace('.JK', '')
                df = create_features(df)
                df = create_label(df)
                
                # Merge dengan Konteks IHSG
                df = df.merge(ihsg_context, left_on='Date', right_index=True, how='left')
                all_data.append(df)
            except:
                continue
        
        # Jeda 2 detik antar chunk agar tidak diblokir Yahoo
        time.sleep(2)
        
    except Exception as e:
        print(f"Gagal di chunk {i}: {e}")
        continue

if not all_data:
    send_telegram("⚠️ Gagal mendownload data. Server Yahoo mungkin sibuk. Coba lagi nanti.")
    exit(0)

df_all = pd.concat(all_data, ignore_index=True).dropna(subset=['rsi', 'target', 'ihsg_trend'])

# =========================
# MODEL TRAINING (No Change)
# =========================
features = [
    'ret_1','ret_3','trend','ma_ratio','vol_ratio','volatility','range',
    'breakout', 'breakout_strength', 'rsi',
    'ihsg_ret_3', 'ihsg_trend'
]

df_all = df_all.sort_values('Date')
split_date = df_all['Date'].quantile(0.7)
train_df = df_all[df_all['Date'] <= split_date]

X_train = train_df[features]
y_train = train_df['target'].astype(int)

scale_pos_weight = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-9)
model = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss'
)
model.fit(X_train, y_train)

# =========================
# LIVE SCREENER (CORE)
# =========================
latest_date = df_all['Date'].max()
df_live = df_all[df_all['Date'] == latest_date].copy()
df_live['prob'] = model.predict_proba(df_live[features])[:,1]

# FILTER PROBABILITAS (HATI SISTEM)
df_final = df_live[
    (df_live['prob'] > 0.70) & 
    (df_live['trend'] > 1.0) & 
    (df_live['vol_ratio'] > 1.1) &
    (df_live['rsi'].between(50, 80))
].copy()

if not df_final.empty:
    df_final['score'] = (df_final['prob'] * 0.7 + df_final['vol_ratio'] * 0.3)
    df_final = df_final.sort_values(by='score', ascending=False).head(10)
    
    msg = f"🚀 *IDX PROBABILITY HITS*\n📅 {latest_date.strftime('%Y-%m-%d')}\n"
    msg += f"📊 IHSG: {'Bullish' if ihsg_context.iloc[-1]['ihsg_trend'] > 1 else 'Bearish'}\n\n"
    for _, row in df_final.iterrows():
        msg += f"• *{row['ticker']}*\n  Prob: {row['prob']:.2f} | Vol: {row['vol_ratio']:.1f}\n"
    send_telegram(msg)
else:
    send_telegram(f"✅ *Scan Selesai*\n{latest_date.strftime('%Y-%m-%d')}\nTidak ada saham memenuhi kriteria probabilitas > 0.70.")
