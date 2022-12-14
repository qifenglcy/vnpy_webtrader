import sys
from vnpy_ctastrategy import strategies 
# from strategies.atr_rsi_strategy_tick import AtrRsiStrategyTick
from strategies.Grid_copy_Status_ALLTRADED_copy_copy import GridStrategy
from vnpy_ctastrategy.strategies.double_ma_strategy import DoubleMaStrategy
from vnpy_ctastrategy.strategies.atr_rsi_strategy import   AtrRsiStrategy

import importlib
import pandas as pd

from enum import Enum
from typing import Any, List, Optional
import asyncio
import json
import os
from datetime import datetime, timedelta
from dataclasses import dataclass

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, status, Depends, Query
from fastapi.responses import HTMLResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from jose import jwt, JWTError
from passlib.context import CryptContext

from vnpy.rpc import RpcClient
from vnpy.trader.object import (
    AccountData,
    ContractData,
    OrderData,
    OrderRequest,
    PositionData,
    SubscribeRequest,
    TickData,
    TradeData
)
from vnpy.trader.constant import (
    Exchange,
    Direction,
    OrderType,
    Offset,
)
from vnpy.trader.utility import load_json, get_file_path
from vnpy_spreadtrading.strategies.statistical_arbitrage_strategy import StatisticalArbitrageStrategy
from strategies.idc_statistical_arbitrage_strategy import IdcStatisticalArbitrageStrategy
from vnpy_spreadtrading.backtesting import BacktestingEngine as spread_backtesting_engine
from vnpy_spreadtrading.base import LegData, SpreadData,BacktestingMode
from vnpy_spreadtrading.strategies.statistical_arbitrage_strategy import StatisticalArbitrageStrategy


from time import sleep
from datetime import datetime, time
from logging import INFO

from strategies.Grid_copy_Status_ALLTRADED_copy_copy import GridStrategy
from strategies.tower_strategy import TowerStrategy
from strategies.turtle_signal_strategy import TurtleSignalStrategy
from strategies.double_ma_strategy import DoubleMaStrategy
from ldc_base.self_base import query_bar_from_tqsdk,load_tick_data
from ldc_base.self_base import correlation_time,correlation_measure
# Web??????????????????
SETTING_FILENAME = "web_trader_setting.json"
SETTING_FILEPATH = get_file_path(SETTING_FILENAME)

setting = load_json(SETTING_FILEPATH)
USERNAME = setting["username"]              # ?????????
PASSWORD = setting["password"]              # ??????
REQ_ADDRESS = setting["req_address"]        # ??????????????????
SUB_ADDRESS = setting["sub_address"]        # ??????????????????


SECRET_KEY = "test"                     # ??????????????????
ALGORITHM = "HS256"                     # ????????????
ACCESS_TOKEN_EXPIRE_MINUTES = 30        # ????????????????????????


# ?????????CryptContext????????????????????????
pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")

# FastAPI??????????????????
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# RPC?????????
rpc_client: RpcClient = None


def to_dict(o: dataclass) -> dict:
    """????????????????????????"""
    data = {}
    for k, v in o.__dict__.items():
        if isinstance(v, Enum):
            data[k] = v.value
        elif isinstance(v, datetime):
            data[k] = str(v)
        else:
            data[k] = v
    return data


class Token(BaseModel):
    """????????????"""
    access_token: str
    token_type: str


def authenticate_user(current_username: str, username: str, password: str):
    """????????????"""
    web_username = current_username
    hashed_password = pwd_context.hash(PASSWORD)

    if web_username != username:
        return False

    if not pwd_context.verify(password, hashed_password):
        return False

    return username


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """????????????"""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_access(token: str = Depends(oauth2_scheme)):
    """REST??????"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    if username != USERNAME:
        raise credentials_exception

    return True


# ??????FastAPI??????
# app = FastAPI()
from fastapi import APIRouter
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi import FastAPI, applications
#cdn????????????????????????fastapi.docs???????????????
def swagger_monkey_patch(*args, **kwargs):
    return get_swagger_ui_html(
        *args, **kwargs,
        swagger_js_url='https://cdn.bootcdn.net/ajax/libs/swagger-ui/4.10.3/swagger-ui-bundle.js',
        swagger_css_url='https://cdn.bootcdn.net/ajax/libs/swagger-ui/4.10.3/swagger-ui.css'
    )
applications.get_swagger_ui_html = swagger_monkey_patch
backtestRouter=APIRouter()
app=backtestRouter

###############################################################################################################
import traceback
def handle_exception(request):
  def wrapper(*args, **kwargs):
    try:
      result = request(*args, **kwargs)
      
    except Exception:# as e:
      result = traceback.format_exc()
    # print('result',result)
    return result
  return wrapper

class Model(BaseModel):
    strategies_name: str = 'DoubleMaStrategy'
    vt_symbol: str = "rb.SHFE"
    interval :str = '1m'
    start: str= '2020-01-02 00:00:00' #cta_bar ????????????2022.1.10-2021.3.4
    end: str = '2020-02-02 00:00:00'  #cta_tick ????????????2020.1.2-2020.1.23
    rate:Optional[float] = 1/10000
    slippage:Optional[float] = 1
    size:Optional[float] = 10
    pricetick:Optional[float] = 1
    capital:Optional[float] = 1_000_000
class parameters(BaseModel):
    fast_window : float = 10
    slow_window : float = 20
    atr_length  : float = 22
    atr_ma_length : float = 10
    rsi_length : float = 5
    rsi_entry : float = 16
    trailing_percent : float = 0.8
    fixed_size : float = 1
# class Model_cta(BaseModel):
#     class_name     : str = 'GridStrategy'
#     strategy_name: str = 'ldc_grid_1'
#     vt_symbol: str = "ru2209.SHFE"
#     setting: dict = {}
# @app.get("/backtest_cta_parameters/{parameters}")
# def backtest_cta_parameters(parameters:str):
#     if parameters=='DoubleMaStrategyTick' or parameters=='DoubleMaStrategy':
#         return json.dumps({"fast_window":10, "slow_window":20})
#     if parameters=='AtrRsiStrategyTick' or parameters=='AtrRsiStrategy':
#         return json.dumps({"atr_length":22,"atr_ma_length":10,"rsi_length":5,
#                 "rsi_entry":16,"trailing_percent":0.8,"fixed_size":1})
@app.post("/backtest")
def get_backtest() -> list:
    spread = SpreadData(
        name="rb-Spread",
        legs=[LegData("rb2205.SHFE"), LegData("rb2209.SHFE")],
        variable_symbols={"A": "rb2205.SHFE", "B": "rb2209.SHFE"},
        # price_multipliers={"rb2205.SHFE": 1, "rb2209.SHFE": -1},
        variable_directions={"rb2205.SHFE": 1, "rb2209.SHFE": -1},
        price_formula="A-B",
        trading_multipliers={"rb2205.SHFE": 1, "rb2209.SHFE": -1},
        active_symbol="rb2205.SHFE",
        min_volume=1.0
        )
    engine = spread_backtesting_engine()
    engine.set_parameters(
    spread=spread,                  #????????????
    interval="1m",                  #K?????????
    start=datetime(2022, 1, 1),     #????????????
    end=datetime(2022,3, 26),         #????????????
    rate=0,                                 #????????????   rate=0.5/10000
    slippage=1,                             #????????????   slippage=2.5
    size=300,                               #????????????   size=300
    pricetick=0,                            #????????????   pricetick=2.5
    capital=1_000_000,              #????????????   capital=30_000_000
    # mode=BacktestingMode.TICK,
)
    engine.add_strategy(StatisticalArbitrageStrategy, {})
    engine.load_data() #1?????????????????????2?????????spread_bar

    engine.run_backtesting() #??????????????????????????????
    calculate_reult = engine.calculate_result()   
    statistics = engine.calculate_statistics(calculate_reult)
    calculate_reult = {'lendatabar':len(engine.history_data),
                    'balance':{str(k): calculate_reult['balance'].to_dict()[k] for k in calculate_reult['balance'].to_dict()},\
	                    'drawdown':{str(k): calculate_reult['drawdown'].to_dict()[k] for k in calculate_reult['drawdown'].to_dict()},\
	                    'net_pnl':{str(k): calculate_reult['net_pnl'].to_dict()[k] for k in calculate_reult['net_pnl'].to_dict()}}
    trades_list = []
    for i,j  in engine.trades.items():
        trades_list.append (j.__dict__)
    # orders_list = engine.limit_orders
    # orders_list = []
    # for i,j  in engine.limit_orders.items():
        # orders_list.append (j.__dict__)
    daily_results=[]
    for i,j in engine.daily_results.items():
        daily_results.append (j.__dict__)
    history_data=[]
    for i in engine.history_data:
        history_data.append([i.datetime,i.open_price,i.close_price,i.high_price,i.low_price,i.volume])
    out={'lennnnnnn':len(engine.history_data),
        'trades':trades_list,
        # 'orders' : orders_list,
        'daily_results':daily_results,
        'calculate_reult':calculate_reult,
        'statistics':statistics,
        'history_data':history_data}
    return json.dumps(out, default=str)
@app.post("/backtest_spread")
def get_backtest_spread(model:Model) -> list:
    
    spread = SpreadData(
        name="rb-Spread",
        legs=[LegData("rb2205.SHFE"), LegData("rb2209.SHFE")],
        variable_symbols={"A": "rb2205.SHFE", "B": "rb2209.SHFE"},
        # price_multipliers={"rb2205.SHFE": 1, "rb2209.SHFE": -1},
        variable_directions={"rb2205.SHFE": 1, "rb2209.SHFE": -1},
        price_formula="A-B",
        trading_multipliers={"rb2205.SHFE": 1, "rb2209.SHFE": -1},
        active_symbol="rb2205.SHFE",
        min_volume=1.0
        )
    if model.interval == 'tick':mode=BacktestingMode.TICK
    else:mode=BacktestingMode.BAR
    engine = spread_backtesting_engine()
    engine.set_parameters(
        spread=spread,                  #????????????
        interval=model.interval,
        start=datetime.strptime(model.start,"%Y-%m-%d %H:%M:%S"), 
        end=datetime.strptime(model.end,"%Y-%m-%d %H:%M:%S"),
        rate=float(model.rate),            #????????????
        slippage=float(model.slippage),    #????????????
        size=float(model.size),            #????????????
        pricetick=float(model.pricetick),  #????????????
        capital=float(model.capital),      #????????????
        mode=mode,
)   
    engine.add_strategy(IdcStatisticalArbitrageStrategy, {})
    if mode==BacktestingMode.BAR:
        engine.load_data() #1?????????????????????2?????????spread_bar
    else:engine.history_data=load_tick_data(engine.spread,engine.start,engine.end,engine.pricetick,'')
    # print(mode,len(engine.history_data))
    if engine.history_data is None:
        return "please check you backtest date"
    if len(engine.history_data)==0:
        return json.dumps({'message':'no related data'})
    engine.run_backtesting()
    calculate_reult = engine.calculate_result()  
    if calculate_reult is None:
        return json.dumps({'message':'not enough data'})
    calculate_reult = engine.calculate_result()   
    statistics = engine.calculate_statistics(calculate_reult)
    calculate_reult = {'lendatabar':len(engine.history_data),
                    'balance':{str(k): calculate_reult['balance'].to_dict()[k] for k in calculate_reult['balance'].to_dict()},\
	                    'drawdown':{str(k): calculate_reult['drawdown'].to_dict()[k] for k in calculate_reult['drawdown'].to_dict()},\
	                    'net_pnl':{str(k): calculate_reult['net_pnl'].to_dict()[k] for k in calculate_reult['net_pnl'].to_dict()}}
    trades_list = []
    for i,j  in engine.trades.items():
        trades_list.append (j.__dict__)
    daily_results=[]
    for i,j in engine.daily_results.items():
        daily_results.append (j.__dict__)
    history_data=[]
    for i in engine.history_data:
        if mode==BacktestingMode.BAR:
            history_data.append([i.datetime,i.open_price,i.close_price,i.high_price,i.low_price,i.volume])
        else:history_data.append([i.datetime,i.open_price,i.last_price,i.high_price,i.low_price,i.volume])
    out={'lennnnnnn':len(engine.history_data),
        'trades':trades_list,
        # 'orders' : orders_list,
         'daily_results':daily_results,
         'calculate_reult':calculate_reult,
         'statistics':statistics,
         'history_data':history_data}
    return json.dumps(out, default=str)
# @app.post("/backtest_cta_bar")
def backtest_cta_bar(model:Model,parameters:parameters) -> list:
    
    engine = cta_backtest()
    engine.set_parameters(
        vt_symbol=model.vt_symbol,
        interval=model.interval,
        start=datetime.strptime(model.start,"%Y-%m-%d %H:%M:%S"),
        end=datetime.strptime(model.end,"%Y-%m-%d %H:%M:%S"),
        rate=float(model.rate),            #????????????
        slippage=float(model.slippage),    #????????????
        size=float(model.size),            #????????????
        pricetick=float(model.pricetick),  #????????????
        capital=float(model.capital),      #????????????
        # mode=BacktestingMode.TICK,
    )
    setting = parameters
    engine.add_strategy(eval(model.strategies_name), setting)
    # engine.load_data()
    # engine.history_data=query_bar_from_tqsdk(engine)
    engine.history_data = query_bar_from_tqsdk(engine)
    if engine.history_data is None:
        return "please check you backtest date"
    if len(engine.history_data)==0:
        return 'no related data'
    engine.run_backtesting()
    calculate_reult = engine.calculate_result()  
    if calculate_reult is None:
        return 'not enough data'
    statistics = engine.calculate_statistics(calculate_reult)
    calculate_reult = {'lendatabar':len(engine.history_data),
                    'balance':{str(k): calculate_reult['balance'].to_dict()[k] for k in calculate_reult['balance'].to_dict()},\
	                    'drawdown':{str(k): calculate_reult['drawdown'].to_dict()[k] for k in calculate_reult['drawdown'].to_dict()},\
	                    'net_pnl':{str(k): calculate_reult['net_pnl'].to_dict()[k] for k in calculate_reult['net_pnl'].to_dict()}}
    trades_list = []
    for i,j  in engine.trades.items():
        trades_list.append (j.__dict__)
    # orders_list = engine.limit_orders
    orders_list = []
    for i,j  in engine.limit_orders.items():
        orders_list.append (j.__dict__)
    daily_results=[]
    for i,j in engine.daily_results.items():
        daily_results.append (j.__dict__)
    history_data=[]
    for i in engine.history_data:
        history_data.append([i.datetime,i.open_price,i.close_price,i.high_price,i.low_price,i.volume])
    out={'lennnnnnn':len(engine.history_data),
        'trades':trades_list,
        'orders' : orders_list,
        'daily_results':daily_results,
        'calculate_reult':calculate_reult,
        'statistics':statistics,
        'history_data':history_data}
    return json.dumps(out, default=str)
from fastapi import File, UploadFile
#???????????????
# from fastapi.responses import FileResponse
# @app.get('/files/{file_name}')
# async def download_file(file_name:str=''):
# 	file_path=file_name
#     return FileResponse(path=file_path,filename=file_name)
# @app.post("/uploadfiles/")
# async def create_upload_file(files: List[UploadFile] = File(...)):
#     for file in files:
#         res = await file.read()#??????????????????
#         with open(file.filename, "wb+") as f:#????????????????????????
#             f.write(res)
#     return {"message": "success"}
###################################################################################################################
# @handle_exception
# file_prefix = r''
file_prefix = r'C:\Users\qifeng'
@app.post("/upload",summary='????????????')
def upload(file:UploadFile):
    # # print('file',file)
    # print('file2',file.read())
    # print('file3',file.file.read())
    csvbates = file.file.read()
    csvname =  file.filename
    # print('csvname',csvname)
    # csvbase64 = base64.b64decode(csvbates)
    import os
    #print('xxxxxxxxxxxxxxxxxxxxxxxx',os.getcwd())
    file_tile = r'\vnpy_web\correlation_time_space_upload\%(csvname)s'%{'csvname':csvname}
    file_path=file_prefix +file_tile
    with open (file_path,'wb+') as f :
        f.write(csvbates)
    # return file_path[1:]#??????????????????????????????
    return file_tile


# @handle_exception
@app.get("/correlation_measure_time_space",summary='???????????????')
def correlation_measure_time_space(serival: int = 12,corr : float=0,file_path:str=r'\vnpy_web\correlation_time_space_upload\BATS_SPY, 1.csv'):
    file_path=file_prefix + r'\%(file_path)s'%{'file_path':file_path}
    # print('2222222222222222',file_path)
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):interval_=pd.read_excel(file_path,index_col=0).dropna()
    elif file_path.endswith('.csv'):interval_=pd.read_csv(file_path,index_col=0,encoding='gbk').dropna()
    else: return 'please upload file with correct format'
    # print('file_path',file_path)
    # interval_=pd.read_csv(r"C:\Users\qifeng\Downloads\BATS_SPY, 1_63e2b.csv",encoding='gbk')
    # interval_=interval_[['A:SPY','B:QQQ','time','A:SPY.1','B:QQQ.1']]
    # a= interval_['A:SPY.1']
    # b= interval_['B:QQQ.1']
    # t= interval_['time']
    # x_total_time=pd.DataFrame({'a':a,'b':b})
    # x_total_data = pd.DataFrame({'t':t,'a':a,'b':b}).dropna()
    # interval_a=interval_[['time','A:SPY.1']].dropna()#.iloc[100:150,:]
    # interval_b=interval_[['time','B:QQQ.1']].dropna()#.iloc[100:150,:]
    interval_['time']=interval_['time'].astype(str)
    # interval_['time'].strftime('%Y-%m-%d %H:%M:%S')
    interval_['time'] = pd.to_datetime(interval_['time'], format='%Y-%m-%d %H:%M:%S')
    interval_a=interval_.iloc[:,[0,1]].dropna()#.iloc[100:150,:]
    interval_b=interval_.iloc[:,[0,2]].dropna()#.iloc[100:150,:]
    # print('interval',interval_a[:10],interval_b[:10])
    corr_measure=round(interval_a.iloc[:,1].corr(interval_b.iloc[:,1]),2)#????????????????????????
    if corr_measure<corr:
        return f'????????????{corr_measure} ??????????????????{corr}??????'
    #????????????????????????
    # interval_a['A:SPY.1']=pd.Series(interval_a['A:SPY.1'][::-1].values)
    # interval_b['B:QQQ.1']=pd.Series(interval_b['B:QQQ.1'][::-1].values)
    #?????????????????????
    x_total_time=pd.concat([interval_a.iloc[:,1],interval_b.iloc[:,1]],axis=1)
    brobability_a_time,brobability_b_time=correlation_measure(x_total_time)
    # if interval_a
    time_range_list_a,time_start_end_a,price_start_end_a,tend_a=correlation_time(interval_a,serival)
    time_range_list_b,time_start_end_b,price_start_end_b,tend_b=correlation_time(interval_b,serival)
    if tend_a!=tend_b:#return '?????????????????????????????????'
        return json.dumps({'data':interval_.values.tolist(),'corr_measure':corr_measure,\
            'brobability_time':{'a':brobability_a_time,'b':brobability_b_time},\
                "log_error":"?????????????????????????????????,???????????????????????????"},default=str)
    time_range_a=pd.concat([pd.Series(i['time_interval'].values) for i in  time_range_list_a],axis=1)
    time_range_b=pd.concat([pd.Series(i['time_interval'].values) for i in  time_range_list_b],axis=1)
    list_brobability=[]
    # brobability_a_analyze,brobability_b_analyze={},{}
    brobability_a_analyze,brobability_b_analyze=[],[]
    for i in time_range_a.keys():
        data_concat_space=pd.concat([time_range_a[i],time_range_b[i]],axis=1)
        data_concat_space.columns=[0,1]
        per_correlation_measure=correlation_measure(data_concat_space)
        list_brobability.append(per_correlation_measure)
        brobability_a_analyze.append([time_start_end_a[i],price_start_end_a[i],per_correlation_measure[0]])
        brobability_b_analyze.append([time_start_end_b[i],price_start_end_b[i],per_correlation_measure[1]])
    tend_up_up_key,tend_up_down_key=['S-H','L-H','L-E'],['S-L','H-E'] #???????????? ['H-L']
    tend_down_down_key,tend_down_up_key=['S-L','H-L','H-E'],['S-H','L-E'] #????????????['L-H']
    file_name='%(a)s_%(b)s_%(start)s_%(end)s' %{'a':interval_a.columns[1],'b':interval_b.columns[1],\
        'start':interval_a.iloc[:,0].min(),'end':interval_a.iloc[:,0].max()}
    file_name=r'/static/correlation_time_space_download/'+file_name.replace('/','-').replace('.','-').replace(':','-')+'.csv'
    file_path=file_prefix+r'/vnpy_front'+file_name
    import os
    if os.path.exists(file_path):os.remove(file_path)
    if tend_a:
        for i in range(0,3):
            time_range_list_a[:3][i][['range',interval_a.columns[1], interval_a.columns[0],'time_interval']].to_csv(file_path,index_label=tend_up_up_key[i]+'/average'+str(serival),mode='a')
            time_range_list_b[:3][i][['range',interval_b.columns[1], interval_b.columns[0],'time_interval']].to_csv(file_path,index_label=tend_up_up_key[i]+'/average'+str(serival),mode='a')
        for i in range(0,2):
            time_range_list_a[3:][i][['range',interval_a.columns[1], interval_a.columns[0],'time_interval']].to_csv(file_path,index_label=tend_up_down_key[i]+'/average'+str(serival),mode='a')
            time_range_list_b[3:][i][['range',interval_b.columns[1], interval_b.columns[0],'time_interval']].to_csv(file_path,index_label=tend_up_down_key[i]+'/average'+str(serival),mode='a')
        brobability_a_up=sum(i[0] for i in list_brobability[:3])/3
        brobability_b_up=sum(i[1] for i in list_brobability[:3])/3
        brobability_a_down=sum(i[0] for i in list_brobability[3:])/2
        brobability_b_down=sum(i[1] for i in list_brobability[3:])/2
        brobability_a_analyze_up= dict(zip(tend_up_up_key,brobability_a_analyze[:3]))
        brobability_b_analyze_up=dict(zip(tend_up_up_key,brobability_b_analyze[:3])) 
        brobability_a_analyze_down= dict(zip(tend_up_down_key,brobability_a_analyze[3:]))
        brobability_b_analyze_down= dict(zip(tend_up_down_key,brobability_b_analyze[3:])) 
        brobability_a_analyze_down['L-H'],brobability_b_analyze_down['L-H']=None,None
    else:
        for i in range(0,3):
            time_range_list_a[:3][i][['range',interval_a.columns[1], interval_a.columns[0],'time_interval']].to_csv(file_path,index_label=tend_down_down_key[i]+'/average'+str(serival),mode='a')
            time_range_list_b[:3][i][['range',interval_b.columns[1], interval_b.columns[0],'time_interval']].to_csv(file_path,index_label=tend_down_down_key[i]+'/average'+str(serival,),mode='a')
        for i in range(0,2):
            time_range_list_a[3:][i][['range',interval_a.columns[1], interval_a.columns[0],'time_interval']].to_csv(file_path,index_label=tend_down_up_key[i]+'/average'+str(serival),mode='a')
            time_range_list_b[3:][i][['range',interval_b.columns[1], interval_b.columns[0],'time_interval']].to_csv(file_path,index_label=tend_down_up_key[i]+'/average'+str(serival),mode='a')
        # [i[['range',i.columns[1], i.columns[0],'time_interval']].to_csv('space_a_b',index_label=tend_down_down_key[i]+' '+str(serival)) for i in time_range_list_a[:3]]
        # [i[['range',i.columns[1], i.columns[0],'time_interval']].to_csv('space_a_b',index_label=tend_down_down_key[i]+' '+str(serival)) for i in time_range_list_b[:3]]
        # [i[['range',i.columns[1], i.columns[0],'time_interval']].to_csv('space_a_b',index_label=tend_down_up_key[i]+' '+str(serival)) for i in time_range_list_a[3:]]
        # [i[['range',i.columns[1], i.columns[0],'time_interval']].to_csv('space_a_b',index_label=tend_down_up_key[i]+' '+str(serival)) for i in time_range_list_b[3:]]
        brobability_a_down=sum(i[0] for i in list_brobability[:3])/3
        brobability_b_down=sum(i[1] for i in list_brobability[:3])/3
        brobability_a_up=sum(i[0] for i in list_brobability[3:])/2
        brobability_b_up=sum(i[1] for i in list_brobability[3:])/2
        brobability_a_analyze_down= dict(zip(tend_down_down_key,brobability_a_analyze[:3]))
        brobability_b_analyze_down=dict(zip(tend_down_down_key,brobability_b_analyze[:3]))
        brobability_a_analyze_up= dict(zip(tend_down_up_key,brobability_a_analyze[3:]))
        brobability_b_analyze_up= dict(zip(tend_down_up_key,brobability_b_analyze[3:]))
        brobability_a_analyze_up['L-H'],brobability_b_analyze_up['L-H']=None,None
# a?????????????????????????????????
    return json.dumps({'data':interval_.values.tolist(),'corr_measure':corr_measure,\
            'brobability_time':{'a':brobability_a_time,'b':brobability_b_time},\
                'brobability_a_up':brobability_a_up,\
                   'brobability_b_up':brobability_b_up,
                   'brobability_a_down':brobability_a_down,\
                       'brobability_b_down':brobability_b_down,\
                           'brobability_a_analyze_up':brobability_a_analyze_up,\
                               'brobability_b_analyze_up':brobability_b_analyze_up,\
                                   'brobability_a_analyze_down':brobability_a_analyze_down,\
                                       'brobability_b_analyze_down':brobability_b_analyze_down,\
                                                'tend':tend_a,'len':len(interval_),\
                                                    'file_path':file_name},default=str)   
#####################################################################################################################
from time import sleep
from datetime import datetime, time
from logging import INFO
from vnpy.event import EventEngine
from vnpy.trader.setting import SETTINGS
from vnpy.trader.engine import MainEngine,LogEngine
#????????????????????????gateway?????????????????????ctpgateway????????????TtsGateway????????????
#???TtsGateway???7*24?????????4097?????????????????????????????????
from vnpy_tts.gateway import TtsGateway
from vnpy_ctp.gateway import CtpGateway
from vnpy_ctastrategy.base import BacktestingMode

'''
@app.post("/tick/{vt_symbol}")
def subscribe(vt_symbol: str, access: bool = Depends(get_access)) -> None:
    """????????????"""
    if not access:
        return "Not authenticated"
    contract: ContractData = rpc_client.get_contract(vt_symbol)
    if not contract:
        return f"???????????????{vt_symbol}"
    req = SubscribeRequest(contract.symbol, contract.exchange)
    rpc_client.subscribe(req, contract.gateway_name)

@app.get("/tick")
def get_all_ticks(access: bool = Depends(get_access)) -> list:
    """??????????????????"""
    if not access:
        return "Not authenticated"
    ticks: List[TickData] = rpc_client.get_all_ticks()
    return [to_dict(tick) for tick in ticks]
class OrderRequestModel(BaseModel):
    """??????????????????"""
    symbol: str
    exchange: Exchange
    direction: Direction
    type: OrderType
    volume: float
    price: float = 0
    offset: Offset = Offset.NONE
    reference: str = ""
@app.post("/order")
def send_order(model: OrderRequestModel, access: bool = Depends(get_access)) -> str:
    """????????????"""
    if not access:
        return "Not authenticated"

    req: OrderRequest = OrderRequest(**model.__dict__)

    contract: ContractData = rpc_client.get_contract(req.vt_symbol)
    if not contract:
        return f"???????????????{req.symbol} {req.exchange.value}"

    vt_orderid: str = rpc_client.send_order(req, contract.gateway_name)
    return vt_orderid


'''

from .vnpy import startup_event,rpc_client,cta_log_data,event_cta_stoporder,event_indicator
print(111111111111111,rpc_client)
rpc_client = startup_event()
print(222222222222222222,rpc_client)
print('rpc_clientrpc_client',rpc_client)
# @handle_exception
@app.get("/cta_strategy/stoporder",summary = '???????????????????????????')
def cta_strategy_stoporder():
    return event_cta_stoporder
@app.get("/cta_strategy/log",summary = '????????????????????????')
def cta_strategy_log():
    return cta_log_data
@app.get("/cta_strategy/strategies",summary='????????????strategies??????')
def cta_strategy_strateties():
    return rpc_client.get_all_strategy_class_names()
# @handle_exception
@app.get("/cta_strategy/strategies/parameters/{class_name}",summary='??????????????????')
def cta_strategy_parameters(class_name:str,trade_mode=''):
    # strategy = eval(parameters)
    parameter= rpc_client.get_strategy_class_parameters(class_name)
    if trade_mode=='cta':
        parameters = {"strategy_name": "", "vt_symbol": ""}
        parameters.update(parameter)
    else:parameters=parameter
    return parameters
@app.get("/cta_strategy/strategy/parameters/{strategy_name}",summary='????????????????????????')
def get_strategy_parameters(strategy_name,trade_mode=''):
    parameter= rpc_client.get_strategy_parameters(strategy_name)
    if trade_mode=='cta':
        parameters = {"strategy_name": "", "vt_symbol": ""}
        parameters.update(parameter)
    else:parameters=parameter
    return parameters
#????????????
class Model_cta(BaseModel):
    class_name     : str = 'GridStrategy'
    strategy_name: str = 'ldc_1'
    vt_symbol: str = "ru2209.SHFE"
    setting:dict= cta_strategy_parameters(class_name)
class Model_backtest(BaseModel):
    class_name: str = 'GridStrategy'
    vt_symbol: str = "ru2209.SHFE"
    interval :str = '1m'
    start: str= '2022-01-01 00:00:00' 
    end: str = '2022-04-01 00:00:00'  
    rate:Optional[float] = 1/10000
    slippage:Optional[float] = 0.2
    size:Optional[float] = 1
    pricetick:Optional[float] = 0.2
    capital:Optional[float] = 1_000_000
    inverse: bool
    setting:dict= cta_strategy_parameters(class_name)
# @handle_exception
@app.get("/cta_strategy/items",summary='????????????????????????')
def get_cta_strategy_items(strategy_name=''):
    sync_data = rpc_client.get_cta_strategy_items(strategy_name)
    return sync_data
# @handle_exception
@app.get("/cta_strategy/strategies/items",summary='????????????????????????')
def get_cta_strategies_items():
    sync_list = rpc_client.get_cta_strategies_items()
    return sync_list
# @handle_exception
@app.post("/cta_strategy/add",summary='????????????')
def add_strategy(model:Model_cta):
    add_startegy_info = rpc_client.add_strategy(model.class_name, model.strategy_name,model.vt_symbol,model.setting)
    print('infoinfo',add_startegy_info,not add_startegy_info)
    if add_startegy_info:
        return add_startegy_info
    else:
        return get_cta_strategy_items(model.strategy_name)
# http://127.0.0.1:8000/items/?skip=0&limit=10
# @handle_exception
@app.get("/cta_strategy/init",summary='???????????????')
def init_strategy(strategy_name: str = 'qf_grid_100',all:str=''):
    if all:
        rpc_client.init_all_strategies()
        sleep(100)
        # print('futuresfutures',{k:v.__dict__ for k,v in futures.items()} )
        return  get_cta_strategies_items()
    rpc_client.init_strategy(strategy_name) #??????????????????inited=True?????????start_strategy????????????False,why
    # print('futuresfuturesfutures',futures)
    # strategies = cta_engine.get('strategies')
    # strategy = strategies.get(strategy_name,'')
    # print('strategy.__class__.__name__ ',strategy.__class__.__name__)
    # if strategy.__class__.__name__ == 'TowerStrategy' :sleep(40)
    # else:sleep(8)
    sleep(20)
    return get_cta_strategy_items(strategy_name)
# inti??????inited=True,??????start??????cta_engine.strategies??????inited=False????????????,????????????????????????strategy???pop??????
# @handle_exception
@app.get("/cta_strategy/start",summary='????????????')
def start_strategy( strategy_name: str,all:str=''):
    if all:
        rpc_client.start_all_strategies()
        return  get_cta_strategies_items()
    rpc_client.start_strategy(strategy_name)
    return get_cta_strategy_items(strategy_name)
# @handle_exception
@app.get("/cta_strategy/stop",summary='????????????')
def stop_strategy( strategy_name: str,all:str=''):
    if all:
        rpc_client.stop_all_strategies()
        return  get_cta_strategies_items()
    rpc_client.stop_strategy(strategy_name)
    return get_cta_strategy_items(strategy_name)
# @app.get("/cta_strategy/cut",summary='????????????')
def cut_strategy(strategy_name:str,save_orderid=[],flag='????????????'):
    rpc_client.cut_strategy(strategy_name)
    return get_cta_strategy_items(strategy_name)
# @handle_exception
@app.post("/cta_strategy/edit",summary='????????????')
def edit_strategy(model:Model_cta):
    rpc_client.edit_strategy(model.strategy_name,model.setting)
    return get_cta_strategy_items(model.strategy_name)
# @handle_exception
@app.get("/cta_strategy/remove",summary='????????????')
def remove_strategy(strategy_name: str):
    rpc_client.remove_strategy(strategy_name)
    return get_cta_strategy_items(strategy_name)
@app.post("/tick/{vt_symbol}")
def subscribe(vt_symbol: str) -> None:
    """????????????"""
    # if not access:
    #     return "Not authenticated"

    contract: ContractData = rpc_client.get_contract(vt_symbol)
    if not contract:
        return f"???????????????{vt_symbol}"

    req = SubscribeRequest(contract.symbol, contract.exchange)
    rpc_client.subscribe(req, contract.gateway_name)


@app.get("/ectalog")
def get_all_ticks(access: bool = Depends(get_access)) -> list:
    """??????????????????"""
    if not access:
        return "Not authenticated"

    ticks: List[TickData] = rpc_client.get_all_ticks()
    return [to_dict(tick) for tick in ticks]
   
# @handle_exception
@app.post("/backtest_cta",summary='CTA??????')
def backtest_cta(model:Model_backtest):
    
    model.start=datetime.strptime(model.start,"%Y-%m-%d %H:%M:%S")
    model.end=datetime.strptime(model.end,"%Y-%m-%d %H:%M:%S")
    bool = rpc_client.start_backtesting(**model.__dict__)
    print((model.end - model.start).days//3)
    print('boolboolboolboolboolboolboolboolboolboolboolbool',bool)
    sleep((model.end - model.start).days//3)
    # if not  bool:return '????????????'
    history_data = rpc_client.get_history_data()
    print('=====================history_data=====================',len(history_data))
    if not history_data or len(history_data)==0:
        return json.dumps({'message':'no related data'})
    calculate_reult = rpc_client.get_result_df()  
    print('=====================calculate_reult=====================',calculate_reult)
    if calculate_reult is None:
        return json.dumps({'message':'not enough data'})
    statistics = rpc_client.get_result_statistics()
    calculate_reult = {'lendatabar':len(history_data),
                    'balance':{str(k): calculate_reult['balance'].to_dict()[k] for k in calculate_reult['balance'].to_dict()},\
	                    'drawdown':{str(k): calculate_reult['drawdown'].to_dict()[k] for k in calculate_reult['drawdown'].to_dict()},\
	                    'net_pnl':{str(k): calculate_reult['net_pnl'].to_dict()[k] for k in calculate_reult['net_pnl'].to_dict()}}
    # daily_results=[]
    # get_all_daily_results=rpc_client.get_all_daily_results()
    # for i,j in get_all_daily_results.items():
    #     daily_results.append (j.__dict__)
    daily_results=rpc_client.get_all_daily_results()
    # trades_list = []
    # trades = rpc_client.get_all_trades()
    # for i,j  in trades.items():
    #     trades_list.append (j.__dict__)
    trades_list = rpc_client.get_all_trades()
    # orders_list = []
    # limit_orders = rpc_client.get_all_orders()
    # for i,j  in limit_orders.items():
    #     orders_list.append (j.__dict__)
    orders_list = rpc_client.get_all_orders()
    history_data_list=[]
    for i in history_data:
        history_data_list.append([i.datetime,i.open_price,i.close_price,i.high_price,i.low_price,i.volume])
    out={'lennnnnnn':len(history_data_list),
        'trades':trades_list,
        'orders' : orders_list,
         'calculate_reult':calculate_reult,
         'statistics':statistics,
         'daily_results':daily_results,
         'history_data':history_data_list}
    return json.dumps(out, default=str)
@app.post("/start_indicator",summary='????????????????????????')
def start_indicator(parameter):
    symbol,indicator,n_kline = parameter.split('.')
    n_kline = 1+'m'
    symbol = symbol+'BINANCE'
    rpc_client.start_indicator(symbol,indicator,n_kline)
@app.post("/view_indicator",summary='??????????????????')
def get_indicator(symbol):
    for i in reversed(event_indicator):
        if symbol ==i.symbol.split('.')[0]:
            return dict([(key,event_indicator[key]) for key in ['symbol','datetime','rsi_ma','label']])
