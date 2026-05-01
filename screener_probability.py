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

# DAFTAR 250+ EMITEN PAPAN UTAMA
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
    'BVIC.JK', 'BWPT.JK', 'BYAN.JK', 'CARS.JK', 'CASA.JK', 'CEKA.JK', 'CENT.JK', 'CFIN.JK', 'CINT.JK', 'CITA.JK',
    'CITY.JK', 'CLAY.JK', 'CLEO.JK', 'CLPI.JK', 'CMNP.JK', 'CNKO.JK', 'CNTX.JK', 'CPIN.JK', 'CSAP.JK', 'CSIS.JK',
    'CTRA.JK', 'DART.JK', 'DAYA.JK', 'DEFI.JK', 'DEWA.JK', 'DGIK.JK', 'DGNS.JK', 'DILD.JK', 'DIVA.JK', 'DKFT.JK',
    'DLTA.JK', 'DMAS.JK', 'DMMX.JK', 'DMND.JK', 'DOID.JK', 'DPNS.JK', 'DSFI.JK', 'DSNG.JK', 'DSSP.JK', 'DUTI.JK',
    'DYAN.JK', 'EAST.JK', 'ECII.JK', 'EKAD.JK', 'ELSA.JK', 'ELTY.JK', 'EMTK.JK', 'ENRG.JK', 'ERAA.JK', 'ESIP.JK',
    'ESSA.JK', 'ESTI.JK', 'EXCL.JK', 'FAPA.JK', 'FAST.JK', 'FISH.JK', 'FORU.JK', 'FPNI.JK', 'FREN.JK', 'GAMA.JK',
    'GDST.JK', 'GDYR.JK', 'GEMA.JK', 'GEMS.JK', 'GGRM.JK', 'GIAA.JK', 'GJTL.JK', 'GLOB.JK', 'GMCW.JK', 'GMTD.JK',
    'GOLD.JK', 'GOOD.JK', 'GOTO.JK', 'GPRA.JK', 'GSMF.JK', 'GWSA.JK', 'GZCO.JK', 'HEAL.JK', 'HELI.JK', 'HERO.JK',
    'HEXA.JK', 'HITS.JK', 'HMSP.JK', 'HOKI.JK', 'HOME.JK', 'HOTL.JK', 'HRUM.JK', 'IATA.JK', 'IBST.JK', 'ICBP.JK',
    'ICON.JK', 'IDPR.JK', 'IGAR.JK', 'IIKP.JK', 'IKAI.JK', 'IKAN.JK', 'IMAS.JK', 'IMJS.JK', 'IMPC.JK', 'INAF.JK',
    'INAI.JK', 'INCF.JK', 'INCO.JK', 'INDF.JK', 'INDS.JK', 'INDX.JK', 'INDY.JK', 'INKP.JK', 'INPC.JK', 'INPP.JK',
    'INPS.JK', 'INRU.JK', 'INTD.JK', 'INTP.JK', 'IPCC.JK', 'IPCM.JK', 'IPOL.JK', 'IRRA.JK', 'ISAT.JK', 'ISIG.JK',
    'ITMA.JK', 'ITMG.JK', 'JECC.JK', 'JIHD.JK', 'JKON.JK', 'JKSW.JK', 'JPFA.JK', 'JRPT.JK', 'JSMR.JK', 'JSPT.JK',
    'JTPE.JK', 'KAEF.JK', 'KAYU.JK', 'KBAG.JK', 'KBLI.JK', 'KBLM.JK', 'KBLV.JK', 'KBRI.JK', 'KDSI.JK', 'KEJU.JK',
    'KIAS.JK', 'KICI.JK', 'KIJA.JK', 'KINO.JK', 'KIOS.JK', 'KJEN.JK', 'KLBF.JK', 'KMDS.JK', 'KMTR.JK', 'KOBX.JK',
    'KOIN.JK', 'KONI.JK', 'KOPI.JK', 'KOTA.JK', 'KPAS.JK', 'KPIG.JK', 'KRAH.JK', 'KRAS.JK', 'KREN.JK', 'LPPS.JK',
    'LPPF.JK', 'LSIP.JK', 'MAIN.JK', 'MAPI.JK', 'MAPA.JK', 'MBMA.JK', 'MDKA.JK', 'MEDC.JK', 'MIKA.JK', 'MNCN.JK',
    'MPPA.JK', 'MYOR.JK', 'PGAS.JK', 'PGEO.JK', 'PNBN.JK', 'PTBA.JK', 'PTPP.JK', 'PWON.JK', 'RAJA.JK',
    'SCMA.JK', 'SIDO.JK', 'SILO.JK', 'SMGR.JK', 'SMRA.JK', 'TINS.JK', 'TKIM.JK', 'TLKM.JK', 'TOWR.JK', 'TPIA.JK',
    'UNTR.JK', 'UNVR.JK', 'VICI.JK', 'WIKA.JK', 'WSKT.JK'
]

os.makedirs("output", exist_ok=True)

def send_telegram(message):
    token = os.getenv('TELEGRAM_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    if not token or not chat_id: return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try: requests.post(url, json=payload)
    except: pass

def create_features(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.ffill()
    df['ret_1'] = df['Close'].pct_change()
    df['ret_3'] = df['Close'].pct_change(3)
    df['ma20'] = df['Close'].rolling(20).mean()
    df['ma50'] = df['Close'].rolling(50).mean()
    df['trend'] = df['Close'] / (df['ma50'] + 1e-9)
    df['ma_gap'] = (df['ma20'] - df['ma50']) / (df['ma50'] + 1e-9)
    df['momentum_slope'] = df['Close'].rolling(5).mean().pct_change(5)
    df['vol_ma'] = df['Volume'].rolling(10).mean()
    df['vol_ratio'] = df['Volume'] / (df['vol_ma'] + 1e-9)
    df['volatility'] = df['Close'].pct_change().rolling(10).std()
    df['range'] = (df['High'] - df['Low']) / (df['Close'] + 1e-9)
    df['hh_10'] = df['High'].rolling(10).max()
    df['breakout'] = df['Close'] / (df['hh_10'] + 1e-9)
    
    # FITUR YANG KEMBALI DIMASUKKAN
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

# DOWNLOAD PROCESS
print(f"Downloading data for {len(TICKERS_IDX)} tickers...")
raw_data = yf.download(TICKERS_IDX, period="1y", interval="1d", group_by='ticker', progress=False, threads=True)

all_data = []
for ticker in TICKERS_IDX:
    try:
        df = raw_data[ticker].copy() if len(TICKERS_IDX) > 1 else raw_data.copy()
        df = df.dropna(subset=['Close'])
        if len(df) < MIN_DATA: continue
        
        df = df.reset_index()
        df['ticker'] = ticker.replace('.JK', '')
        df = create_features(df)
        df = create_label(df)
        all_data.append(df)
    except:
        continue

if not all_data:
    send_telegram("❌ Error: Gagal mendownload data.")
    exit(0)

df_all = pd.concat(all_data, ignore_index=True).dropna(subset=['rsi', 'target', 'breakout_strength'])

# MACHINE LEARNING
features = ['ret_1','ret_3','trend','ma_gap','momentum_slope','vol_ratio','volatility','range','breakout', 'breakout_strength', 'rsi']
df_all = df_all.sort_values('Date')
split_date = df_all['Date'].quantile(0.7)
train_df = df_all[df_all['Date'] <= split_date]
test_df  = df_all[df_all['Date'] > split_date]

X_train = train_df[features]
y_train = train_df['target'].astype(int)

scale_pos_weight = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-9)
model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss')
model.fit(X_train, y_train)

# LIVE SCANNING
latest_date = df_all['Date'].max()
df_live = df_all[df_all['Date'] == latest_date].copy()
df_live['prob'] = model.predict_proba(df_live[features])[:,1]

df_final = df_live[
    (df_live['prob'] > 0.70) & 
    (df_live['trend'] > 1.0) & 
    (df_live['vol_ratio'] > 1.2)
].copy()

if not df_final.empty:
    df_final['score'] = (df_final['prob'] * 0.7 + df_final['vol_ratio'] * 0.3)
    df_final = df_final.sort_values(by='score', ascending=False).head(10)
    
    msg = f"🚀 *IDX MAIN BOARD HITS*\n📅 {latest_date.strftime('%Y-%m-%d')}\n\n"
    for _, row in df_final.iterrows():
        msg += f"• *{row['ticker']}*\n  Prob: {row['prob']:.2f} | Strength: {row['breakout_strength']:.1f}\n  Vol Ratio: {row['vol_ratio']:.1f} | RSI: {row['rsi']:.1f}\n\n"
    send_telegram(msg)
else:
    send_telegram(f"✅ *Screener Selesai*\n{latest_date.strftime('%Y-%m-%d')}\nTidak ada sinyal.")
