import re,sys
from symtable import Symbol
from fastapi import APIRouter

extendRouter=APIRouter()


from vnpy.trader.optimize import OptimizationSetting


sys.path.append(r'C:\Users\Administrator')
from vnpy_ctastrategy.strategies.double_ma_strategy import DoubleMaStrategy
from vnpy_ctastrategy.strategies.atr_rsi_strategy import   AtrRsiStrategy
from base.self_base import load_data_hdf5_cta,query_bar_from_tqsdk,get_trade_record,calculate_result_with_trade_record
from datetime import datetime
from pydantic import BaseModel


from fastapi import APIRouter
extendRouter=APIRouter()
class calcWithRecordRes(BaseModel):
    date: str
    show_profit: float
    holding_pnl: float
    trading_pnl: float
    total_pnl: float

class Item(BaseModel):
    symbol: str 
    exchange: str 
    start_time: str 
    end_time: str 
    


# @extendRouter.post("/calcWithRecord", response_model=calcWithRecordRes)
# def calculate_result_with_trade(pdata:calcWithRecordReq):
@extendRouter.post("/calcWithRecord")
def calculate_result_with_trade(pdata:Item):
    """
    从deribit获取成交记录 得出分析统计结果\n
    
    self.holding_pnl = self.start_pos * \(self.close_price - self.pre_close) * size\n

    self.trading_pnl += pos_change * \(self.close_price - trade.price) * size\n

    self.total_pnl = self.trading_pnl + self.holding_pnl\n

    self.net_pnl = self.total_pnl - self.commission - self.slippage\n
    """
    # vt_symbol="y2209.DCE",
    # interval="1m",
    # start=datetime(2022, 4, 2),
    # end=datetime(2022, 4, 3),
    # rate=0,
    # slippage=0,
    # size=1,
    # pricetick=0.5,
    # capital=1
    allTrade=get_trade_record(pdata)
    df = calculate_result_with_trade_record(allTrade)
    return df

 