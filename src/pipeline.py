"""Pipeline P15: UCI + Open-Meteo, PostgreSQL e regressão temporal."""
from pathlib import Path
import argparse, hashlib, io, json, os, zipfile
import joblib
import numpy as np
import pandas as pd
import requests
from sqlalchemy import create_engine, text
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
UCI_URL = 'https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip'
WEATHER_URL = 'https://archive-api.open-meteo.com/v1/archive'
PARAMS = dict(latitude=38.9072, longitude=-77.0369, start_date='2011-01-01', end_date='2012-12-31', daily='temperature_2m_mean,precipitation_sum,wind_speed_10m_max', timezone='America/New_York', models='era5')
NUM = ['temperature_2m_mean','precipitation_sum','wind_speed_10m_max','trend','month_sin','month_cos']
CAT = ['weekday','holiday','workingday']

def extract(refresh=False):
    raw=ROOT/'data/raw'; raw.mkdir(parents=True,exist_ok=True)
    for name,url,params in [('uci.zip',UCI_URL,None),('weather.json',WEATHER_URL,PARAMS)]:
        path=raw/name
        if refresh or not path.exists():
            r=requests.get(url,params=params,timeout=90); r.raise_for_status()
            if name.endswith('.json'): r.json()
            path.write_bytes(r.content)
    with zipfile.ZipFile(raw/'uci.zip') as z:
        name=next(n for n in z.namelist() if n.endswith('day.csv'))
        bikes=pd.read_csv(io.BytesIO(z.read(name)))
    weather=pd.DataFrame(json.loads((raw/'weather.json').read_text())['daily']).rename(columns={'time':'dteday'})
    bikes=bikes[['dteday','holiday','workingday','cnt']].copy()
    for frame in [bikes,weather]:
        frame['dteday']=pd.to_datetime(frame['dteday']).dt.date
        if frame['dteday'].duplicated().any(): raise ValueError('Data duplicada na fonte')
        if frame.isna().any().any(): raise ValueError('Fonte contém valores ausentes')
    if (bikes['cnt']<0).any() or (weather['precipitation_sum']<0).any(): raise ValueError('Valores negativos inválidos')
    manifest={'sources':[{'name':'UCI Bike Sharing','url':UCI_URL},{'name':'Open-Meteo ERA5','url':WEATHER_URL,'params':PARAMS}], 'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [raw/'uci.zip', raw/'weather.json']}}
    (raw/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    return bikes,weather

def integrate(bikes,weather,offline=False):
    if offline:
        data=bikes.merge(weather,on='dteday',validate='one_to_one')
    else:
        url=os.environ.get('DATABASE_URL')
        if not url: raise RuntimeError('Defina DATABASE_URL. --offline é somente verificação local sem PostgreSQL.')
        engine=create_engine(url)
        with engine.begin() as conn:
            for stmt in (ROOT/'sql/schema.sql').read_text().split(';'):
                if stmt.strip(): conn.execute(text(stmt))
            conn.execute(text('TRUNCATE TABLE bike_daily, weather_daily'))
            bikes.to_sql('bike_daily',conn,if_exists='append',index=False)
            weather.to_sql('weather_daily',conn,if_exists='append',index=False)
            data=pd.read_sql(text('SELECT b.*, w.temperature_2m_mean, w.precipitation_sum, w.wind_speed_10m_max FROM bike_daily b JOIN weather_daily w USING (dteday) ORDER BY dteday'),conn)
    if len(data)!=len(bikes) or len(data)!=len(weather): raise ValueError('Integração perdeu datas')
    data=data.sort_values('dteday').reset_index(drop=True)
    date=pd.to_datetime(data.dteday)
    data['weekday']=date.dt.dayofweek
    data['trend']=(date-pd.Timestamp('2011-01-01')).dt.days
    data['month_sin']=np.sin(2*np.pi*date.dt.month/12)
    data['month_cos']=np.cos(2*np.pi*date.dt.month/12)
    data.to_csv(ROOT/'data/processed/integrated.csv',index=False)
    return data

def train(data,offline=False):
    cut=int(len(data)*.8); train=data.iloc[:cut]; test=data.iloc[cut:]
    pre=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),NUM),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CAT)])
    models={'Ridge':Ridge(alpha=10),'RandomForest':RandomForestRegressor(n_estimators=200,min_samples_leaf=3,random_state=42,n_jobs=1),'GradientBoosting':GradientBoostingRegressor(n_estimators=150,max_depth=2,learning_rate=.05,random_state=42)}
    rows=[]; fitted={}
    for name,model in models.items():
        pipe=Pipeline([('preprocess',pre),('model',model)])
        scores=-cross_val_score(pipe,train[NUM+CAT],train.cnt,cv=TimeSeriesSplit(n_splits=4),scoring='neg_root_mean_squared_error',n_jobs=1)
        pipe.fit(train[NUM+CAT],train.cnt); fitted[name]=pipe
        rows.append({'model':name,'cv_rmse_mean':float(scores.mean()),'cv_rmse_std':float(scores.std())})
    # Escolha exclusivamente pela validação no treino, antes de olhar o teste.
    best=min(rows,key=lambda r:r['cv_rmse_mean'])['model']
    baseline=DummyRegressor(strategy='mean').fit(train[NUM+CAT],train.cnt)
    for name,model in {**fitted,'BaselineMean':baseline}.items():
        pred=model.predict(test[NUM+CAT])
        record=next((r for r in rows if r['model']==name),None)
        if record is None: record={'model':name,'cv_rmse_mean':None,'cv_rmse_std':None}; rows.append(record)
        record.update(mae=float(mean_absolute_error(test.cnt,pred)),rmse=float(mean_squared_error(test.cnt,pred)**.5),r2=float(r2_score(test.cnt,pred)))
    pd.DataFrame(rows).to_csv(ROOT/'results/metrics.csv',index=False)
    pred=fitted[best].predict(test[NUM+CAT])
    pd.DataFrame({'date':test.dteday,'actual':test.cnt,'prediction':pred,'residual':test.cnt-pred}).to_csv(ROOT/'results/predictions.csv',index=False)
    joblib.dump(fitted[best],ROOT/'models/best_model.pkl')
    summary={'selected_model':best,'selection':'menor RMSE de validação temporal no treino','rows':len(data),'train_rows':len(train),'test_rows':len(test),'train_start':str(train.dteday.iloc[0]),'train_end':str(train.dteday.iloc[-1]),'test_start':str(test.dteday.iloc[0]),'test_end':str(test.dteday.iloc[-1]),'postgresql_executed':not offline,'features':NUM+CAT,'limitations':['Clima observado do próprio dia: análise retrospectiva condicional, não previsão antecipada operacional.','Reanálise ERA5 em um ponto representa aproximadamente a área de Washington DC.','Apenas dois anos e uma cidade; não generalizar a Macapá sem validação local.','casual e registered excluídos pois somam a variável alvo cnt.']}
    (ROOT/'results/summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    plt.figure(figsize=(10,4)); plt.plot(pd.to_datetime(test.dteday),test.cnt,label='Observado'); plt.plot(pd.to_datetime(test.dteday),pred,label=best);plt.ylabel('Aluguéis por dia');plt.legend();plt.tight_layout();plt.savefig(ROOT/'results/predictions.png',dpi=150);plt.close()
    plt.figure(figsize=(10,3));plt.plot(pd.to_datetime(data.dteday),data.cnt);plt.ylabel('Aluguéis por dia');plt.tight_layout();plt.savefig(ROOT/'results/demand.png',dpi=150);plt.close()
    print(json.dumps(summary,ensure_ascii=False,indent=2));print(pd.DataFrame(rows).to_string(index=False))
    return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--offline',action='store_true');p.add_argument('--refresh',action='store_true');args=p.parse_args()
    for folder in ['data/processed','models','results']: (ROOT/folder).mkdir(parents=True,exist_ok=True)
    bikes,weather=extract(args.refresh);data=integrate(bikes,weather,args.offline);train(data,args.offline)
if __name__=='__main__': main()
