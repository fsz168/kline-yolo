#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K线形态识别扫描系统 v6.0
基于yfinance + 自研形态识别引擎
支持: A股/港股/美股 ETF和股票批量扫描
17种形态，完全对齐《量化指标解码19》PDF规则
"""
import pandas as pd, numpy as np, yfinance as yf, matplotlib, warnings, os, json
from datetime import datetime
from collections import Counter
warnings.filterwarnings('ignore')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
os.environ['HOME'] = '/root'

# ============================================================
# 用户配置区
# ============================================================
TARGET_POOL = 'etf'       # 'etf' | 'a_stock' | 'hk_stock' | 'us_stock'
OUTPUT_DIR  = '/root/kline-yolo/scans'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ETF标的池
ETF_POOL = {
    '510050.SS':'50ETF',    '510300.SS':'300ETF',   '510500.SS':'500ETF',
    '159915.SZ':'创业板ETF',  '159919.SZ':'深证100',   '510100.SS':'上证指数ETF',
    '588000.SS':'科创50ETF',  '588080.SS':'科创50ETF2','510330.SS':'沪深300ETF',
    '159901.SZ':'深证ETF',
    '512480.SS':'半导体ETF',  '512760.SS':'芯片ETF',    '515050.SS':'5G ETF',
    '515980.SS':'人工智能',   '515700.SS':'新能源车',   '512170.SS':'医疗ETF',
    '159928.SZ':'消费ETF',   '512690.SS':'酒ETF',     '512010.SS':'医药ETF',
    '512660.SS':'军工ETF',   '159869.SZ':'机器人ETF', '159865.SZ':'游戏ETF',
    '515220.SS':'煤炭ETF',   '515790.SS':'光伏ETF',
    '513050.SS':'中概互联网', '159605.SZ':'中概互联2',  '159792.SZ':'游戏ETF2',
    '513500.SS':'纳指ETF',   '513100.SS':'纳指100',   '513300.SS':'标普500',
    '159941.SZ':'纳指ETF2',  '513080.SS':'港股通ETF', '159509.SZ':'港股消费',
    '513130.SS':'标普消费',  '159631.SZ':'港股金融',  '513020.SS':'德国DAX',
    '513550.SS':'港股医药',  '159607.SZ':'科创做市',
    '518880.SS':'黄金ETF',   '159934.SZ':'黄金ETF2',
    '512980.SS':'豆粕ETF',   '159985.SZ':'豆粕ETF2',
    '561350.SS':'能源化工',  '159827.SZ':'能源ETF2',
    '511010.SS':'国债ETF',   '511260.SS':'十年国债',  '511030.SS':'企债ETF',
    '511380.SS':'可转债ETF', '511800.SS':'货币ETF',   '511090.SS':'货币ETF2',
    '512000.SS':'券商ETF',   '512190.SS':'证券ETF',   '159805.SZ':'创业板成长',
    '159801.SZ':'芯片设备',  '159992.SZ':'创新药ETF',
}

# A股蓝筹池（示例，可替换为akshare拉取的完整代码）
A_STOCK_POOL = {
    '600519.SS':'贵州茅台', '000858.SZ':'五粮液',   '601318.SS':'中国平安',
    '600036.SS':'招商银行', '601288.SS':'农业银行', '600030.SS':'中信证券',
    '000333.SZ':'美的集团', '002475.SZ':'立讯精密', '300750.SZ':'宁德时代',
    '688981.SS':'中芯国际', '002594.SZ':'比亚迪',   '600276.SS':'恒瑞医药',
}

def get_pool():
    if TARGET_POOL == 'etf': return ETF_POOL
    elif TARGET_POOL == 'a_stock': return A_STOCK_POOL
    return ETF_POOL

# ============================================================
# 辅助指标
# ============================================================
def calc_atr(h,l,c,p=14):
    tr=np.maximum(h-l,np.abs(h-np.roll(c,1)),np.abs(l-np.roll(c,1))); tr[0]=h[0]-l[0]
    return pd.Series(tr).rolling(p).mean().values

def calc_boll(c,p=20,m=2.0):
    mid=pd.Series(c).rolling(p).mean().values; std=pd.Series(c).rolling(p).std().values
    return mid+m*std, mid, mid-m*std

def calc_ma(c,p): return pd.Series(c).rolling(p).mean().values
def calc_vol_ma(vol,p=20): return pd.Series(vol).rolling(p).mean().values

def get_data(code, interval='1wk', count=30):
    try:
        tk=yf.Ticker(code); df=tk.history(period='1y',interval=interval)
        if df is None or len(df)<12: return None
        return df.tail(count).reset_index()
    except: return None

def br(o,c,h,l,i):
    rng=h[i]-l[i]; return abs(c[i]-o[i])/rng if rng>0 else 0

# ============================================================
# 形态检测器（17种）
# ============================================================
def bull_eng(o,h,l,c,i,a):
    if i<1: return False
    if br(o,c,h,l,i)<0.50: return False
    if (h[i]-l[i])<1.2*a[i]: return False
    return (c[i-1]<o[i-1] and c[i]>o[i] and o[i]<=c[i-1] and c[i]>=o[i-1])

def bear_eng(o,h,l,c,i,a):
    if i<1: return False
    if br(o,c,h,l,i)<0.50: return False
    if (h[i]-l[i])<1.2*a[i]: return False
    return (c[i-1]>o[i-1] and c[i]<o[i] and c[i]<=o[i-1] and o[i]>=c[i-1])

def morning_star(o,h,l,c,i,a):
    if i<2: return False
    if c[i-2]>=o[i-2] or br(o,c,h,l,i-2)<0.45: return False
    sb=abs(c[i-1]-o[i-1]); sr=h[i-1]-l[i-1]
    if sr>1e-9 and sb/sr>0.55: return False
    return (c[i]>o[i] and c[i]>(o[i-2]+c[i-2])/2)

def evening_star(o,h,l,c,i,a):
    if i<2: return False
    if c[i-2]<=o[i-2] or br(o,c,h,l,i-2)<0.45: return False
    sb=abs(c[i-1]-o[i-1]); sr=h[i-1]-l[i-1]
    if sr>1e-9 and sb/sr>0.55: return False
    return (c[i]<o[i] and c[i]<(o[i-2]+c[i-2])/2)

def piercing(o,h,l,c,i):
    if i<1: return False
    if br(o,c,h,l,i)<0.50: return False
    return (c[i-1]<o[i-1] and c[i]>o[i] and c[i]>(o[i-1]+c[i-1])/2 and c[i]<o[i-1])

def dark_cloud(o,h,l,c,i):
    if i<1: return False
    if br(o,c,h,l,i)<0.50: return False
    return (c[i-1]>o[i-1] and c[i]<o[i] and c[i]<(o[i-1]+c[i-1])/2 and c[i]>o[i-1])

def hammer(o,h,l,c,i,a):
    if (h[i]-l[i])<1.0*a[i]: return False
    lower=min(o[i],c[i])-l[i]; rng=h[i]-l[i]; body=abs(c[i]-o[i])
    if body/rng>0.35: return False
    return lower>rng*0.55 and lower>2.0*body

def shooting_star(o,h,l,c,i,a):
    if (h[i]-l[i])<1.0*a[i]: return False
    upper=h[i]-max(o[i],c[i]); rng=h[i]-l[i]; body=abs(c[i]-o[i])
    if body/rng>0.35: return False
    return upper>rng*0.55 and upper>2.0*body

def three_inside_up(o,h,l,c,i,a):
    if i<2: return False
    if c[i-2]>o[i-2]: return False
    if not (h[i-1]<=h[i-2] and l[i-1]>=l[i-2]): return False
    return (c[i]>o[i] and br(o,c,h,l,i)>=0.65 and c[i]>(o[i-2]+c[i-2])/2)

def three_inside_down(o,h,l,c,i,a):
    if i<2: return False
    if c[i-2]<o[i-2]: return False
    if not (h[i-1]<=h[i-2] and l[i-1]>=l[i-2]): return False
    return (c[i]<o[i] and br(o,c,h,l,i)>=0.65 and c[i]<(o[i-2]+c[i-2])/2)

def tweezer_bottom(o,h,l,c,i):
    if i<1: return False
    if abs(l[i]-l[i-1])/(l[i]+1e-9)<0.003:
        return (c[i]>o[i] or c[i-1]>o[i-1])
    return False

def tweezer_top(o,h,l,c,i):
    if i<1: return False
    if abs(h[i]-h[i-1])/(h[i]+1e-9)<0.003:
        return (c[i]<o[i] or c[i-1]<o[i-1])
    return False

def consec_up(o,h,l,c,i,n=3):
    """连续阳线: 允许中间1根小回调(实体<40%)"""
    if i<n-2: return False
    for j in range(i-n+1,i+1):
        if c[j]>=o[j]: continue
        if br(o,c,h,l,j)<0.40: continue
        return False
    return br(o,c,h,l,i)>=0.60

def consec_down(o,h,l,c,i,n=3):
    if i<n-2: return False
    for j in range(i-n+1,i+1):
        if c[j]<=o[j]: continue
        if br(o,c,h,l,j)<0.40: continue
        return False
    return br(o,c,h,l,i)>=0.60

def tight_consol(o,h,l,c,i,win=5):
    if i<win: return False
    rc=c[i-win+1:i+1]; rh=h[i-win+1:i+1]; rl=l[i-win+1:i+1]
    pr=(max(rh)-min(rl))/(np.mean(rc)+1e-9)
    return pr<0.055

def breakout_setup(o,h,l,c,i,a):
    if i<5: return False
    rng4=(h[i-3:i+1].max()-l[i-3:i+1].min())/(np.mean(c[i-3:i+1])+1e-9)
    if rng4>0.07: return False
    if br(o,c,h,l,i-4)<0.50: return False
    return c[i]>h[i-3:i+1].max()

def bull_pullback(o,h,l,c,i,a):
    """上升回踩: 上升趋势中连续2根阴线或出现锤子"""
    if i<2: return False
    if c[i]<o[i] and c[i-1]<o[i-1]: return True
    if hammer(o,h,l,c,i,a): return True
    return False

CHECKERS = {
    'bull_eng':     lambda o,h,l,c,i,a: bull_eng(o,h,l,c,i,a),
    'bear_eng':     lambda o,h,l,c,i,a: bear_eng(o,h,l,c,i,a),
    'morning':      lambda o,h,l,c,i,a: morning_star(o,h,l,c,i,a),
    'evening':      lambda o,h,l,c,i,a: evening_star(o,h,l,c,i,a),
    'piercing':     lambda o,h,l,c,i,a: piercing(o,h,l,c,i),
    'dark_cloud':   lambda o,h,l,c,i,a: dark_cloud(o,h,l,c,i),
    'hammer':       lambda o,h,l,c,i,a: hammer(o,h,l,c,i,a),
    'shooting':     lambda o,h,l,c,i,a: shooting_star(o,h,l,c,i,a),
    '3inside_up':   lambda o,h,l,c,i,a: three_inside_up(o,h,l,c,i,a),
    '3inside_down': lambda o,h,l,c,i,a: three_inside_down(o,h,l,c,i,a),
    'tweezer_bot':  lambda o,h,l,c,i,a: tweezer_bottom(o,h,l,c,i),
    'tweezer_top':  lambda o,h,l,c,i,a: tweezer_top(o,h,l,c,i),
    'consec_up':    lambda o,h,l,c,i,a: consec_up(o,h,l,c,i),
    'consec_down':  lambda o,h,l,c,i,a: consec_down(o,h,l,c,i),
    'tight_consol': lambda o,h,l,c,i,a: tight_consol(o,h,l,c,i),
    'breakout':     lambda o,h,l,c,i,a: breakout_setup(o,h,l,c,i,a),
    'bull_pullback':lambda o,h,l,c,i,a: bull_pullback(o,h,l,c,i,a),
}

PNAMES = {
    'bull_eng':'✅看涨吞没','bear_eng':'❌看跌吞没','morning':'✅晨星',
    'evening':'❌晚星','piercing':'✅刺穿线','dark_cloud':'❌暗云盖顶',
    'hammer':'✅锤子线','shooting':'❌射击之星','3inside_up':'✅三内柱上涨',
    '3inside_down':'❌三内柱下跌','tweezer_bot':'✅平头底','tweezer_top':'❌平头顶',
    'consec_up':'🔥连续阳线','consec_down':'⚡连续阴线',
    'tight_consol':'📊横盘整理','breakout':'🚀突破前兆','bull_pullback':'📊上升回踩',
}

def get_trend(c,ma20):
    if len(c)<20: return 'unknown'
    return 'up' if c[-1]>ma20[-1] else ('down' if c[-1]<ma20[-1] else 'sideways')

def score_sig(ptype, pname, c, o, h, l, ma20, vol, vol_ma, trend):
    base=70
    is_bull = any(x in pname for x in ['✅','🔥','🚀','📊'])
    is_bear = any(x in pname for x in ['❌','⚡'])
    if trend=='up' and is_bull: base+=20
    elif trend=='down' and is_bear: base+=20
    elif trend=='up' and is_bear: base-=35
    elif trend=='down' and is_bull: base-=35
    if '连续' in pname: base+=10
    if any(x in pname for x in ['吞没','晨星','刺穿','锤子']): base+=5
    base+=int(br(o,c,h,l,len(o)-1)*10)
    if len(vol)>=5 and len(vol_ma)>=5:
        vr=np.mean(vol[-3:])/(np.mean(vol_ma[-5:])+1e-9)
        base+=int(min(vr,2.5)*5)
    if trend in ('up','sideways'): base+=5
    return max(0,min(100,base))

def plot_it(code, name, o, h, l, c, vol, patterns, score, trend, save_path):
    n=len(o)
    atr=calc_atr(h,l,c); ma20=calc_ma(c,20)
    upper,mb,lower=calc_boll(c)
    C_UP='#c0392b'; C_DN='#27ae60'; C_MA='#e67e22'; C_BL='#3498db'
    fig=plt.figure(figsize=(6.4,6.4),facecolor='white')
    ax=fig.add_axes([0,0,1,1]); ax.set_facecolor('white')
    for i in range(n):
        col=C_UP if c[i]>=o[i] else C_DN
        ax.add_patch(plt.Rectangle((i-0.4,min(o[i],c[i])),0.8,max(abs(c[i]-o[i]),1e-9),
            facecolor=col,edgecolor=col,lw=0,zorder=3))
        ax.plot([i,i],[max(o[i],c[i]),h[i]],color=col,lw=0.8,zorder=4)
        ax.plot([i,i],[l[i],min(o[i],c[i])],color=col,lw=0.8,zorder=4)
    x=np.arange(n)
    v=~np.isnan(ma20); ax.plot(x[v],ma20[v],color=C_MA,lw=1.2,alpha=0.9,zorder=5)
    vb=~(np.isnan(upper)|np.isnan(lower))
    if vb.sum()>0:
        ax.plot(x[vb],upper[vb],color=C_BL,lw=0.7,alpha=0.5,ls='--',zorder=5)
        ax.plot(x[vb],lower[vb],color=C_BL,lw=0.7,alpha=0.5,ls='--',zorder=5)
        ax.fill_between(x[vb],upper[vb],lower[vb],alpha=0.04,color=C_BL,zorder=2)
    yr=h.max()-l.min()
    seen=set()
    for pt,[pos],sc in patterns:
        if pt in seen: continue
        seen.add(pt)
        pname=PNAMES.get(pt,pt)
        pos=max(0,min(pos,n-1))
        bp=h[pos]; ly=bp+yr*0.05
        pc=C_UP if any(x in pname for x in ['✅','🔥','🚀']) else (C_DN if any(x in pname for x in ['❌','⚡']) else '#e67e22')
        ax.annotate(pname,xy=(pos,bp),xytext=(pos,ly),fontsize=9,fontweight='bold',
            color=pc,ha='center',va='bottom',
            bbox=dict(boxstyle='round,pad=0.3',facecolor='#FFFDE7',edgecolor=pc,alpha=0.9,lw=1.5),
            arrowprops=dict(arrowstyle='->',color=pc,lw=1.5),zorder=10)
    ax.text(0.5,0.975,f'{code}  {name}  |  {trend.upper()}  |  评分:{score}',
        transform=ax.transAxes,fontsize=10,fontweight='bold',ha='center',va='top',color='#333',
        bbox=dict(boxstyle='round,pad=0.4',facecolor='#E3F2FD',edgecolor='#1565C0',lw=1.5))
    ax.set_xlim(-0.5,n-0.5)
    ax.set_ylim(l.min()-yr*0.06,h.max()+yr*0.18)
    ax.axis('off')
    try:
        fig.savefig(save_path,dpi=100,bbox_inches='tight',pad_inches=0)
        plt.close(fig); return True
    except: plt.close(fig); return False

def run_scan(pool_override=None):
    TS = datetime.now().strftime('%Y%m%d_%H%M%S')
    pool = pool_override or get_pool()
    IMG_DIR = f'{OUTPUT_DIR}/images_{TS}'
    REP_DIR = f'{OUTPUT_DIR}/reports_{TS}'
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(REP_DIR, exist_ok=True)

    print(f'{"="*65}')
    print(f'🚀 K线形态扫描 | {len(pool)}只标的 | {TS}')
    print(f'{"="*65}')
    print('📡 预取数据...')
    
    raw={}
    for idx,(code,name) in enumerate(pool.items()):
        try:
            df=get_data(code)
            if df is not None and len(df)>=15: raw[code]=(name,df)
        except: pass
        if (idx+1)%15==0: print(f'  {idx+1}/{len(pool)} 有效:{len(raw)}')
    print(f'  ✅ 有效:{len(raw)}/{len(pool)}\n')

    all_sig=[]
    for code,(name,df) in raw.items():
        try:
            o=df['Open'].values; h=df['High'].values; l=df['Low'].values
            c=df['Close'].values; vol=df['Volume'].values; n=len(o)
            a=calc_atr(h,l,c); ma20=calc_ma(c,20)
            vol_ma=calc_vol_ma(vol); trend=get_trend(c,ma20)
            found=[]
            for check_i in [n-1,n-2,n-3]:
                if check_i<1: continue
                for pt,ck in CHECKERS.items():
                    try:
                        if ck(o,h,l,c,check_i,a):
                            pname=PNAMES.get(pt,pt)
                            if trend=='up' and pt in ['evening','bear_eng','shooting','3inside_down']:
                                pname='📊上升回踩'
                            sc=score_sig(pt,pname,c,o,h,l,ma20,vol,vol_ma,trend)
                            found.append((pt,[check_i],sc,pname))
                    except: pass
            if found:
                best=max(found,key=lambda x:x[2])
                pt,[pos],sc,pname=best
                all_sig.append({'code':code,'name':name,'pattern':pname,'ptype':pt,
                                 'score':sc,'trend':trend,'o':o,'h':h,'l':l,'c':c,
                                 'vol':vol,'found':[(pt,[pos],sc) for pt,[pos],sc,_ in found]})
        except: pass

    all_sig.sort(key=lambda x:x['score'],reverse=True)

    for s in all_sig:
        ip=f"{IMG_DIR}/{s['code'].replace('.','_')}.png"
        plot_it(s['code'],s['name'],s['o'],s['h'],s['l'],s['c'],s['vol'],
                s['found'],s['score'],s['trend'],ip)

    dist=Counter([s['pattern'] for s in all_sig])
    bulls=[s for s in all_sig if any(x in s['pattern'] for x in ['✅','🔥','🚀','📊']) and '❌' not in s['pattern'] and '⚡' not in s['pattern']]
    bears=[s for s in all_sig if any(x in s['pattern'] for x in ['❌','⚡'])]

    print(f'{"="*65}')
    print(f'📊 完成: {len(all_sig)}/{len(raw)} 检出形态信号')
    print(f'{"="*65}')
    print('\n📈 形态分布:')
    for p,cnt in dist.most_common():
        print(f'  {p:15s}  {cnt:2d}  {"█"*cnt}')
    print(f'\n  🟢看涨(含回踩)={len(bulls)}  🔴看跌(含做空)={len(bears)}')
    print(f'\n🏆 TOP候选池:')
    print(f'{"="*65}')
    print(f'{"#":2s}  {"代码":12s}  {"名称":12s}  {"形态":14s}  {"评分":4s}  {"趋势"}')
    print('-'*65)
    for i,s in enumerate(all_sig[:25]):
        d='🟢' if s in bulls else '🔴'
        print(f'{i+1:2d}. {s["code"]:12s}  {s["name"]:12s}  {s["pattern"]:14s}  {s["score"]:4d}  {s["trend"].upper()}  {d}')

    df_out=pd.DataFrame([{'代码':s['code'],'名称':s['name'],'形态':s['pattern'],
        '综合评分':s['score'],'趋势':s['trend'],
        '方向':'看涨' if s in bulls else '看跌'} for s in all_sig])
    csv_path=f'{REP_DIR}/signals_{TS}.csv'
    df_out.to_csv(csv_path,index=False,encoding='utf-8-sig')
    for s in all_sig:
        for k in ['o','h','l','c','vol','found']: s.pop(k,None)
    with open(f'{REP_DIR}/signals_detail_{TS}.json','w',encoding='utf-8') as f:
        json.dump(all_sig,f,ensure_ascii=False,default=str)
    print(f'\n📁 CSV: {csv_path}')
    print(f'📁 图:   {IMG_DIR}/ ({len(all_sig)}张)')
    return all_sig

if __name__ == '__main__':
    run_scan()
