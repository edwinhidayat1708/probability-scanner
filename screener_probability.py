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
CHUNK_SIZE = 20 # Lebih kecil agar lebih aman dari blokir Yahoo

# LIST 258 TICKERS PAPAN UTAMA (Lengkap)
TICKERS_IDX = [
    'AALI.JK', 'ABMM.JK', 'ACES.JK', 'ADHI.JK', 'ADMF.JK', 'ADMR.JK', 'ADRO.JK', 'AGRO.JK', 'AGRS.JK', 'AHAP.JK',
    'AKRA.JK', 'ALDO.JK', 'ALKA.JK', 'AMAR.JK', 'AMRT.JK', 'ANDI.JK', 'ANJT.JK', 'ANTM.JK', 'APIC.JK', 'APLI.JK',
    'APLN.JK', 'ARCI.JK', 'ARNA.JK', 'ARTA.JK', 'ARTO.JK', 'ASGR.JK', 'ASII.JK', 'ASJT.JK', 'ASMI.JK', 'ASRI.JK',
    'ASRM.JK', 'ASSA.JK', 'ATIC.JK', 'AUTO.JK', 'AVIA.JK', 'AXIO.JK', 'BAPA.JK', 'BATA.JK', 'BBCA.JK', 'BBHI.JK',
    'BBKP.JK', 'BBLD.JK', 'BBMD.JK', 'BBNI.JK', 'BBRI.JK', 'BBSP.JK', 'BBTN.JK', 'BBYB.JK', 'BCAP.JK', 'BCIC.JK',
    'BDMN.JK', 'BEBS.JK', 'BEEF.JK', 'BEKS.JK', 'BELI.JK', 'BFIN.JK', 'BGTG.JK', 'BHIT.JK', 'BIPI.JK', 'BIRD.JK',
    'BISI.JK', 'BJBR.JK', 'BJTM.JK', 'BKDP.JK', 'BKSL.JK', 'BVIC.JK', 'BMAS.JK', 'BMRI.JK', 'BMTR.JK', 'BNGA.JK',
    'BNII.JK', 'BNLI.JK', 'BOBA.JK', 'BOLA.JK', 'BPII.JK', 'BPTR.JK', 'BRIS.JK', 'BRMS.JK', 'BRPT.JK', 'BSDE.JK',
    'BSIM.JK', 'BSSR.JK', 'BSWD.JK', 'BTEK.JK', 'BTPS.JK', 'BUKA.JK', 'BUKK.JK', 'BULL.JK', 'BUMI.JK', 'BUVA.JK',
    'BWPT.JK', 'BYAN.JK', 'CARS.JK', 'CASA.JK', 'CEKA.JK', 'CENT.JK', 'CFIN.JK', 'CINT.JK', 'CITA.JK', 'CITY.JK',
    'CLAY.JK', 'CLEO.JK', 'CLPI.JK', 'CMNP.JK', 'CNKO.JK', 'CNTX.JK', 'CPIN.JK', 'CSAP.JK', 'CSIS.JK', 'CTRA.JK',
    'DART.JK', 'DAYA.JK', 'DEFI.JK', 'DEWA.JK', 'DGIK.JK', 'DGNS.JK', 'DILD.JK', 'DIVA.JK', 'DKFT.JK', 'DLTA.JK',
    'DMAS.JK', 'DMMX.JK', 'DMND.JK', 'DOID.JK', 'DPNS.JK', 'DSFI.JK', 'DSNG.JK', 'DSSP.JK', 'DUTI.JK', 'DYAN.JK',
    'EAST.JK', 'ECII.JK', 'EKAD.JK', 'ELSA.JK', 'ELTY.JK', 'EMTK.JK', 'ENRG.JK', 'ERAA.JK', 'ESIP.JK', 'ESSA.JK',
    'ESTI.JK', 'EXCL.JK', 'FAPA.JK', 'FAST.JK', 'FISH.JK', 'FORU.JK', 'FPNI.JK', 'FREN.JK', 'GAMA.JK', 'GDST.JK',
    'GDYR.JK', 'GEMA.JK', 'GEMS.JK', 'GGRM.JK', 'GIAA.JK', 'GJTL.JK', 'GLOB.JK', 'GMCW.JK', 'GMTD.JK', 'GOLD.JK',
    'GOOD.JK', 'GOTO.JK', 'GPRA.JK', 'GSMF.JK', 'GWSA.JK', 'GZCO.JK', 'HEAL.JK', 'HELI.JK', 'HERO.JK', 'HEXA.JK',
    'HITS.JK', 'HMSP.JK', 'HOKI.JK', 'HOME.JK', 'HOTL.JK', 'HRUM.JK', 'IATA.JK', 'IBST.JK', 'ICBP.JK', 'ICON.JK',
    'IDPR.JK', 'IGAR.JK', 'IIKP.JK', 'IKAI.JK', 'IKAN.JK', 'IMAS.JK', 'IMJS.JK', 'IMPC.JK', 'INAF.JK', 'INAI.JK',
    'INCF.JK', 'INCO.JK', 'INDF.JK', 'INDS.JK', 'INDX.JK', 'INDY.JK', 'INKP.JK', 'INPC.JK', 'INPP.JK', 'INPS.JK',
    'INRU.JK', 'INTD.JK', 'INTP.JK', 'IPCC.JK', 'IPCM.JK', 'IPOL.JK', 'IRRA.JK', 'ISAT.JK', 'ISIG.JK', 'ITMA.JK',
    'ITMG.JK', 'JECC.JK', 'JIHD.JK', 'JKON.JK', 'JKSW.JK', 'JPFA.JK', 'JRPT.JK', 'JSMR.JK', 'JSPT.JK', 'JTPE.JK',
    'KAEF.JK', 'KAYU.JK', 'KBAG.JK', 'KBLI.JK', 'KBLM.JK', 'KBLV.JK', 'KBRI.JK', 'KDSI.JK', 'KEJU.JK', 'KIAS.JK',
    'KICI.JK', 'KIJA.JK', 'KINO.JK', 'KIOS.JK', 'KJEN.JK', 'KLBF.JK', 'KMDS.JK', 'KMTR.JK', 'KOBX.JK', 'KOIN.JK',
    'KONI.JK', 'KOPI.JK', 'KOTA.JK', 'KPAS.JK', 'KPIG.JK', 'KRAH.JK', 'KRAS.JK', 'KREN.JK', 'LPPS.JK', 'LPPF.JK',
    'LSIP.JK', 'MAIN.JK', 'MAPI.JK', 'MAPA.JK', 'MBMA.JK', 'MDKA.JK', 'MEDC.JK', 'MIKA.JK', 'MNCN.JK', 'MPPA.JK',
    'MYOR.JK', 'PGAS.JK', 'PGEO.JK', 'PNBN.JK', 'PTBA.JK', 'PTPP.JK', 'PWON.JK', 'RAJA.JK', 'SCMA.JK', 'SIDO.JK',
    'SILO.JK', 'SMGR.JK', 'SMRA.JK', 'TINS.JK', 'TKIM.JK', 'TLKM.JK', 'TOWR.JK', 'TPIA.JK', 'UNTR.JK', 'UNVR.JK',
    'VICI.JK', 'WIKA.JK', 'WSKT.JK'
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
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.ffill()
    # Logic Feature Utama
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
    # RSI
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
# DATA COLLECTION
# ==========================================
print("Mengambil data IHSG...")
ihsg_data = yf.download('^JKSE', period="1y", interval="1d", progress=False)
ihsg_context = pd.DataFrame(index=ihsg_data.index)
ihsg_context['ihsg_ret_3'] = ihsg_data['Close'].pct_change(3)
ihsg_context['ihsg_trend'] = ihsg_data['Close'] / ihsg_data['Close'].rolling(50).mean()
ihsg_context = ihsg_context.ffill()

all_data = []
for i in range(0, len(TICKERS_IDX), CHUNK_SIZE):
    chunk = TICKERS_IDX[i:i + CHUNK_SIZE]
    print(f"Proses chunk {i//CHUNK_SIZE + 1}...")
    try:
        data_chunk = yf.download(chunk, period="1y", interval="1d", group_by='ticker', progress=False)
        for ticker in chunk:
            try:
                df = data_chunk[ticker].copy() if len(chunk) > 1 else data_chunk.copy()
                df = df.dropna(subset=['Close'])
                if len(df) < MIN_DATA or df['Volume'].tail(5).mean() < 500: continue # Filter likuiditas
                
                df = df.reset_index()
                df['ticker'] = ticker.replace('.JK', '')
                df = create_features(df)
                df = create_label(df)
                df = df.merge(ihsg_context, left_on='Date', right_index=True, how='left')
                all_data.append(df)
            except: continue
        time.sleep(1.5) # Jeda antar chunk
    except: continue

if not all_data:
    send_telegram("⚠️ Gagal mengambil data market.")
    exit(0)

df_all = pd.concat(all_data, ignore_index=True).dropna(subset=['rsi', 'target', 'ihsg_trend'])

# =========================
# MODEL TRAINING
# =========================
features = ['ret_1','ret_3','trend','ma_ratio','vol_ratio','volatility','range','breakout', 'breakout_strength', 'rsi', 'ihsg_ret_3', 'ihsg_trend']
df_all = df_all.sort_values('Date')
split_date = df_all['Date'].quantile(0.7)
train_df = df_all[df_all['Date'] <= split_date]
X_train = train_df[features]
y_train = train_df['target'].astype(int)

scale_pos_weight = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-9)
model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss')
model.fit(X_train, y_train)

# =========================
# OUTPUT & NOTIFICATION
# =========================
latest_date = df_all['Date'].max()
df_live = df_all[df_all['Date'] == latest_date].copy()
df_live['prob'] = model.predict_proba(df_live[features])[:,1]

# FILTER UTAMA
df_final = df_live[(df_live['prob'] > 0.70) & (df_live['trend'] > 1.0) & (df_live['vol_ratio'] > 1.1)].copy()

if not df_final.empty:
    df_final = df_final.sort_values(by='prob', ascending=False).head(10)
    msg = f"🚀 *IDX PROBABILITY HITS (258 Tickers)*\n📅 {latest_date.strftime('%Y-%m-%d')}\n"
    msg += f"📊 IHSG: {'Bullish' if ihsg_context.iloc[-1]['ihsg_trend'] > 1 else 'Bearish'}\n\n"
    for _, row in df_final.iterrows():
        msg += f"• *{row['ticker']}*\n  Prob: {row['prob']:.2f} | RSI: {row['rsi']:.1f}\n"
    send_telegram(msg)
else:
    send_telegram(f"✅ *Screener Selesai ({latest_date.strftime('%Y-%m-%d')})*\nTidak ada saham dari 258 emiten yang menembus probabilitas > 0.70 hari ini.")
