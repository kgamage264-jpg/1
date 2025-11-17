# main.py
import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objs as go
import requests
import time

# ---------------------------
# Indicator functions
# ---------------------------
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def true_range(df):
    prev_close = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (df['low'] - prev_close).abs()
    return pd.concat([tr1,tr2,tr3],axis=1).max(axis=1)

def atr(df, period=14):
    tr = true_range(df)
    return tr.rolling(window=period, min_periods=1).mean()

def rsi(df, period=14):
    delta = df['close'].diff()
    up = delta.where(delta>0,0.0)
    down = -delta.where(delta<0,0.0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up/(ma_down+1e-9)
    return 100 - (100/(1+rs))

# ---------------------------
# Fetch data
# ---------------------------
def get_exchange():
    return ccxt.binance({'enableRateLimit': True})

def fetch_ohlcv(exchange,symbol,timeframe,limit=500):
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
    return df

# ---------------------------
# Generate signals
# ---------------------------
def generate_signal(df, params):
    if len(df) < max(params['ema_short'], params['ema_long'], params['rsi_period']):
        return None
    df = df.copy()
    df['ema_short'] = ema(df['close'], params['ema_short'])
    df['ema_long'] = ema(df['close'], params['ema_long'])
    df['rsi'] = rsi(df, params['rsi_period'])
    df['atr'] = atr(df, params['atr_period'])

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last

    signal='neutral'
    reason=[]
    confidence=0.0

    # EMA crossover logic
    if (prev['ema_short'] <= prev['ema_long']) and (last['ema_short']>last['ema_long']):
        if last['rsi']<params['rsi_overbought']:
            signal='long'
            confidence+=0.6
            reason.append('EMA bullish crossover')
    elif (prev['ema_short']>=prev['ema_long']) and (last['ema_short']<last['ema_long']):
        if last['rsi']>params['rsi_oversold']:
            signal='short'
            confidence+=0.6
            reason.append('EMA bearish crossover')
    else:
        reason.append('No EMA crossover')

    # Momentum confirmation
    if signal=='long' and last['close']>prev['high']:
        confidence+=0.2
        reason.append('Breakout above previous high')
    elif signal=='short' and last['close']<prev['low']:
        confidence+=0.2
        reason.append('Breakdown below previous low')

    # Entry, TP, SL
    if signal in ('long','short'):
        atr_mul=params['atr_multiplier']
        if signal=='long':
            entry=last['close']
            sl=entry-last['atr']*atr_mul
            tp=entry+last['atr']*atr_mul*params['tp_atr_factor']
        else:
            entry=last['close']
            sl=entry+last['atr']*atr_mul
            tp=entry-last['atr']*atr_mul*params['tp_atr_factor']
        return {
            'signal': signal,
            'entry': round(entry, params['price_precision']),
            'tp': round(tp, params['price_precision']),
            'sl': round(sl, params['price_precision']),
            'confidence': round(min(confidence,1.0),2),
            'rsi': round(last['rsi'],1),
            'atr': round(last['atr'],8),
            'reason': '; '.join(reason)
        }
    else:
        return {
            'signal':'neutral',
            'confidence':round(confidence,2),
            'rsi':round(last['rsi'],1),
            'atr':round(last['atr'],8),
            'reason':'; '.join(reason)
        }

# ---------------------------
# Gemini AI explanation (optional, free)
# ---------------------------
def gemini_explain(signal_data, hf_token=None):
    if not hf_token:
        return "No AI token — free signals only"
    try:
        prompt=f"""
        Explain this crypto signal simply:
        Signal: {signal_data['signal']}
        Entry: {signal_data.get('entry','-')}
        TP: {signal_data.get('tp','-')}
        SL: {signal_data.get('sl','-')}
        RSI: {signal_data.get('rsi','-')}
        Reason: {signal_data.get('reason','-')}
        """
        url="https://api-inference.huggingface.co/models/google/flan-t5-large"
        headers={"Authorization": f"Bearer {hf_token}"}
        payload={"inputs": prompt}
        response=requests.post(url, headers=headers, json=payload, timeout=15)
        if response.status_code==200:
            text=response.json()
            if isinstance(text,list) and len(text)>0:
                return text[0].get('generated_text','No explanation')
        return "No AI explanation"
    except Exception as e:
        return f"AI error: {str(e)}"

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Crypto Hybrid Bot", layout="wide")
st.title("Hybrid Crypto Signal Bot — Free + Gemini AI Explanation")

# Sidebar
symbol=st.sidebar.text_input("Symbol (Binance)","BTC/USDT")
timeframes=st.sidebar.multiselect("Timeframes",["1m","3m","5m","15m","30m","1h"],default=["1m","5m","15m"])
max_bars=st.sidebar.number_input("History bars",500,50,2000,50)
price_precision=st.sidebar.number_input("Price decimals",2,0,8)
update_interval=st.sidebar.slider("Refresh (sec)",10,600,30)

hf_token=st.sidebar.text_input("HuggingFace token (optional)", type="password")

st.sidebar.markdown("### Strategy Parameters")
ema_short=st.sidebar.number_input("EMA Short",9,2,200)
ema_long=st.sidebar.number_input("EMA Long",21,2,400)
rsi_period=st.sidebar.number_input("RSI Period",14,2,50)
rsi_overbought=st.sidebar.number_input("RSI Overbought",70)
rsi_oversold=st.sidebar.number_input("RSI Oversold",30)
atr_period=st.sidebar.number_input("ATR Period",14)
atr_multiplier=st.sidebar.number_input("ATR multiplier for SL",1.5,0.1,10.0)
tp_atr_factor=st.sidebar.number_input("TP factor",2.0,0.1,10.0)

params={
    'ema_short':int(ema_short),
    'ema_long':int(ema_long),
    'rsi_period':int(rsi_period),
    'rsi_overbought':float(rsi_overbought),
    'rsi_oversold':float(rsi_oversold),
    'atr_period':int(atr_period),
    'atr_multiplier':float(atr_multiplier),
    'tp_atr_factor':float(tp_atr_factor),
    'price_precision':int(price_precision)
}

exchange=get_exchange()
placeholder=st.empty()

# Auto-refresh
from streamlit_autorefresh import st_autorefresh
count = st_autorefresh(interval=update_interval*1000, limit=None, key="auto")

with placeholder.container():
    results=[]
    errors=[]
    for tf in timeframes:
        try:
            df=fetch_ohlcv(exchange,symbol,tf,max_bars)
            sig=generate_signal(df,params)
            row={
                'timeframe':tf,
                'signal':sig.get('signal','neutral'),
                'entry':sig.get('entry',None),
                'tp':sig.get('tp',None),
                'sl':sig.get('sl',None),
                'confidence':sig.get('confidence',None),
                'rsi':sig.get('rsi',None),
                'reason':sig.get('reason','')
            }
            row['ai_explanation']=gemini_explain(row,hf_token)
            results.append(row)
        except Exception as e:
            errors.append(f"{tf}: {repr(e)}")
            results.append({'timeframe':tf,'signal':'error','entry':None,'tp':None,'sl':None,'confidence':None,'rsi':None,'reason':'Error','ai_explanation':'No AI'})

    df_results=pd.DataFrame(results).set_index('timeframe')
    st.subheader(f"Signals for {symbol}")
    st.table(df_results[['signal','entry','tp','sl','confidence','ai_explanation']])

    if errors:
        st.error("Some timeframes failed:")
        for err in errors:
            st.text(err)

# Chart for first timeframe
if len(timeframes)>0:
    try:
        tf_plot=timeframes[0]
        df_plot=fetch_ohlcv(exchange,symbol,tf_plot,200)
        df_plot['ema_short']=ema(df_plot['close'],params['ema_short'])
        df_plot['ema_long']=ema(df_plot['close'],params['ema_long'])
        fig=go.Figure(data=[go.Candlestick(x=df_plot.index, open=df_plot['open'], high=df_plot['high'], low=df_plot['low'], close=df_plot['close'], name='Price')])
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['ema_short'], name=f'EMA{params["ema_short"]}'))
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['ema_long'], name=f'EMA{params["ema_long"]}'))
        fig.update_layout(height=500, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig,use_container_width=True)
    except Exception as e:
        st.write("Chart error:",e)

st.caption("Hybrid Bot: Free EMA+RSI+ATR signals + Gemini AI explanation (HuggingFace free model).")
