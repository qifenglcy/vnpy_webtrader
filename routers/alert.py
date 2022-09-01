import json
from threading import Thread
from time import time
from fastapi import APIRouter

from vnpy.event.engine import EventEngine
from vnpy.trader.constant import Exchange
from vnpy.trader.engine import MainEngine
from vnpy.trader.object import SubscribeRequest
from vnpy_tts import TtsGateway # tts 24h 模拟数据 使用前必须停用ctp


alertRouter=APIRouter()

def async_call(fn):
    def wrapper(*args, **kwargs):
        Thread(target=fn, args=args, kwargs=kwargs).start()

    return wrapper

@alertRouter.post("/setConfig")
def set_config(pdata: list):
    # 最新价预警 价差预警格式
    # pdata = [
    #     {
    #         'id': '1',
    #         'symbol':"y2209",
    #         'exchange':"DCE",
    #         'type':"1",
    #         'price':"12000",
    #         'tag':"last_price",
    #     },
    #     {
    #         'id': '2',
    #         'symbol':"rb2210",
    #         'exchange':"CFFEX",
    #         'type':"0",
    #         'price':"10000",
    #         'tag':"last_price",
    #     },
    #     # {
    #     #     'id': '3',
    #     #     'symbol_A':"y2209",
    #     #     'exchange_A':"DCE",
    #     #     'symbol_B':"rb2210",
    #     #     'exchange_B':"CFFEX",
    #     #     'type':"0",
    #     #     'price':"10000",
    #     #     'tag':"spread",
    #     # }
    # ]

    print(pdata)

    f = open('E:/alert.json', 'w+')
    f.write(json.dumps(pdata))
    f.close()

    #异步化后不会堵塞return
    # subscribeSymbol(pdata)

    return {
        'code':"0",
        'msg':'success',
        'data':pdata
    }


# @async_call
def subscribe_symbol(engine:any):

    f = open('/alert.json', 'r+')
    alertList = json.loads(f.read())
    f.close()

    # 取得最新配置去订阅
    for value in alertList:
        engine.subscribe( SubscribeRequest(symbol=value['symbol'],exchange=Exchange(value['exchange'])),"TTS")
    

def check_price(alertList: list, engine:any):
    while True:

        f = open('./alert.json', 'r+')
        alertList = json.loads(f.read())
        f.close()

        ticks = engine.get_all_ticks()

        if len(ticks) > 0:
            for tick in ticks:
                # print(tick.name,tick.symbol, tick.exchange,tick.last_price)

                for value in alertList: 
                    if value['symbol'] == tick.symbol:
                        if  value['type'] == '1' and tick.last_price > float(value['price']):
                            sendEmail(value, f'{value["symbol"]}当前价格{tick.last_price}，大于设定价格{value["price"]}，发送邮件')

                        if  value['type'] == '0' and tick.last_price < float(value['price']):
                            sendEmail(value, f'{value["symbol"]}当前价格{tick.last_price}，小于设定价格{value["price"]}，发送邮件')

alertPool = {}
def sendEmail(item,msg):

    if alertPool.get(item['id']) is None:
        alertPool[item['id']] = time()
        print('发送邮件1', msg, time())
        
    else:
        if time() - alertPool[item['id']] > 60*3:
            alertPool[item['id']] = time()
            print('发送邮件2', msg, time())
        else:
            return
            print('邮件发送过于频繁')
    



 
     
    


