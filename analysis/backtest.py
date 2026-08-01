import ast,json
from pathlib import Path
import numpy as np,pandas as pd,requests,yfinance as yf

START='2010-01-01'; END='2026-08-01'; LAST='20260731'; INIT=125; TACT=10
BUY=0.00015; SELL=0.00215 # 2026 KOSPI tax 0.20% + assumed fee 0.015%

def get_data():
 u=f'https://api.finance.naver.com/siseJson.naver?symbol=000660&requestType=1&startTime=20100101&endTime={LAST}&timeframe=day'
 meta={}; n=y=None
 try:
  r=requests.get(u,timeout=60,headers={'User-Agent':'Mozilla/5.0'});r.raise_for_status();a=ast.literal_eval(r.text.strip())
  n=pd.DataFrame(a[1:],columns=[str(x).strip() for x in a[0]]).rename(columns={'날짜':'date','시가':'open','고가':'high','저가':'low','종가':'close','거래량':'volume'})
  n=n[['date','open','high','low','close','volume']];n.date=pd.to_datetime(n.date.astype(str),format='%Y%m%d')
  for c in n.columns[1:]:n[c]=pd.to_numeric(n[c],errors='coerce')
  n=n.dropna().drop_duplicates('date').set_index('date').sort_index();meta['naver_rows']=len(n)
 except Exception as e:meta['naver_error']=repr(e)
 try:
  y=yf.download('000660.KS',start=START,end=END,auto_adjust=False,progress=False,threads=False)
  if isinstance(y.columns,pd.MultiIndex):y.columns=y.columns.get_level_values(0)
  y=y.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})[['open','high','low','close','volume']].dropna();meta['yahoo_rows']=len(y)
 except Exception as e:meta['yahoo_error']=repr(e)
 if n is None and y is None:raise RuntimeError(meta)
 if n is not None and y is not None:
  z=n[['close']].join(y[['close']],how='inner',lsuffix='_n',rsuffix='_y');q=(z.close_n/z.close_y-1).abs()
  meta.update(overlap=len(z),median_abs_diff_pct=float(q.median()*100),p95_abs_diff_pct=float(q.quantile(.95)*100),last_naver=float(z.close_n.iloc[-1]),last_yahoo=float(z.close_y.iloc[-1]))
 d=n if n is not None and len(n)>500 else y;meta['primary']='Naver' if d is n else 'Yahoo';return d,meta

def ind(d,win=120):
 x=d.copy();x['r5']=x.close.pct_change(5);x['m20']=x.close.rolling(20).mean();x['m60']=x.close.rolling(60).mean();x['m200']=x.close.rolling(200).mean();x['v20']=x.volume.rolling(20).mean();x['vr']=x.volume/x.v20;x['hi']=x.high.rolling(win).max();x['g20']=x.close/x.m20-1;x['nh']=x.close/x.hi-1;return x

def sim(raw,start,end,p=None,cost=1):
 d=ind(raw,(p or {}).get('win',120)).loc[start:end].copy();sp=float(d.open.iloc[0]);ep=float(d.close.iloc[-1]);sh=INIT;cash=0.;lots=[];selln=buyn=cyc=cd=0;eq=[];last=-99
 if p is None:
  v=INIT*d.close;mdd=float((v/v.cummax()-1).min()*100);yrs=(d.index[-1]-d.index[0]).days/365.25
  return dict(name='hold',start=start,end=end,shares=INIT,cash=0,eqshares=INIT,gain=0,value=INIT*ep,excess=0,cycles=0,open=0,sells=0,buys=0,cashdays=0,mdd=mdd,cagr=((ep/sp)**(1/yrs)-1)*100)
 for i,row in enumerate(d.itertuples()):
  eq.append(sh*row.close+cash);cd+=bool(lots)
  if i<200 or i+1>=len(d):continue
  if lots:
   sq=sum(q['q'] for q in lots);avg=sum(q['net'] for q in lots)/sq;no=float(d.open.iloc[i+1]);aff=int(cash//(no*(1+BUY*cost)))
   stable=i>=2 and d.close.iloc[i]>=d.close.iloc[i-1]>=d.close.iloc[i-2]
   near=abs(row.close/row.m20-1)<=p['band'] or abs(row.close/row.m60-1)<=p['band'] or row.close<=row.m20
   trend=row.close>=row.m200 or row.m60>=d.m60.iloc[max(0,i-20)]
   if row.close<=avg*(1-p['drop']) and aff>=sq+1 and stable and near and trend:
    pay=aff*no*(1+BUY*cost);cash-=pay;sh+=aff;buyn+=1;cyc+=1;lots=[];continue
  out=sum(q['q'] for q in lots)
  if TACT-out<5:continue
  c=[row.r5>=p['r5'],row.nh>=-p['near'],row.g20>=p['gap'],row.vr>=p['vr']]
  trend=row.close>=row.m200 and row.m60>=d.m60.iloc[max(0,i-20)]
  if sum(c)>=p['req'] and trend:
   if lots and (i-last<2 or row.close<lots[0]['px']):continue
   no=float(d.open.iloc[i+1]);net=5*no*(1-SELL*cost);sh-=5;cash+=net;lots.append({'q':5,'net':net,'px':no});selln+=1;last=i
 val=sh*ep+cash;bh=INIT*ep;e=val/ep;v=pd.Series(eq,index=d.index[:len(eq)]);yrs=(d.index[-1]-d.index[0]).days/365.25
 return dict(name=p.get('name','rule'),start=start,end=end,shares=sh,cash=round(cash),eqshares=e,gain=e-INIT,value=round(val),excess=(val/bh-1)*100,cycles=cyc,open=int(bool(lots)),sells=selln,buys=buyn,cashdays=cd,mdd=float((v/v.cummax()-1).min()*100),cagr=((val/(INIT*sp))**(1/yrs)-1)*100)

def main():
 d,meta=get_data();d=d.loc[START:'2026-07-31']
 current=dict(name='current_4of4',r5=.25,near=.03,gap=.15,vr=2,req=4,drop=.10,band=.05,win=120)
 moderate=dict(name='moderate_3of4',r5=.20,near=.05,gap=.12,vr=1.5,req=3,drop=.10,band=.05,win=120)
 candidates=[]
 for r5 in [.15,.20,.25]:
  for gap in [.10,.15]:
   for vr in [1.5,2]:
    for req in [3,4]:
     for drop in [.08,.10,.12]:candidates.append(dict(name='grid',r5=r5,near=.05,gap=gap,vr=vr,req=req,drop=drop,band=.05,win=120))
 wins=[('2013-01-01','2016-12-31'),('2017-01-01','2020-12-31'),('2021-01-01','2022-12-31')];rank=[]
 for p in candidates:
  rr=[sim(d,a,b,p) for a,b in wins];g=np.array([x['gain'] for x in rr]);ex=np.array([x['excess'] for x in rr]);op=sum(x['open'] for x in rr);cy=sum(x['cycles'] for x in rr)
  score=g.mean()*2+np.median(g)*1.5+(g>0).sum()-op*1.5+np.minimum(ex,0).mean()*.25-g.std()*.35;rank.append((score,cy,np.median(g),p,rr))
 rank.sort(key=lambda x:x[0],reverse=True);best=next((x[3] for x in rank if x[1]>=2 and x[2]>0),rank[0][3]).copy();best['name']='robust_selected'
 periods={'full':('2013-01-01','2026-07-31'),'oos':('2023-01-01','2026-07-31'),'ai':('2024-01-01','2026-07-31')};res=[]
 for k,(a,b) in periods.items():
  for p in [None,current,moderate,best]:
   z=sim(d,a,b,p);z['period']=k;res.append(z)
 sens=[sim(d,'2023-01-01','2026-07-31',best,c) for c in [.75,1,1.5,2]]
 out=dict(generated=str(pd.Timestamp.utcnow()),data=dict(meta,rows=len(d),first=str(d.index.min().date()),last=str(d.index.max().date())),current=current,moderate=moderate,best=best,results=res,cost_sensitivity=sens,limitations=['Historical daily consensus EPS revisions are not freely reproducible, so EPS-divergence was not backtested.','Signals use close data and execute at next open to avoid look-ahead bias.','Official quarterly results remain a fundamental veto; dividends and loan interest are excluded because they are common to variants.'])
 Path('analysis/output').mkdir(parents=True,exist_ok=True);Path('analysis/output/results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));pd.DataFrame(res).to_csv('analysis/output/summary.csv',index=False)
 print('BACKTEST_JSON_START');print(json.dumps(out,ensure_ascii=False));print('BACKTEST_JSON_END')
if __name__=='__main__':main()
