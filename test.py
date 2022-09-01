import json
import time

import requests


def get_trade_record():
        proxy = {
            'https':'http://127.0.0.1:1080'
        }

        url = "https://test.deribit.com/api/v2/"
        key = "0lbLerQW"
        secret = "H2k16nd4zRNUaSU7BTYTMBpUgTtk28uO1rANHVhNxr0"

        auth = url + 'public/auth?client_id='+key+'&client_secret='+secret+'&grant_type=client_credentials'

        response = requests.get(auth,proxies=proxy,timeout=15)
        response = json.loads(response.text)

        print(response)
        

        headers = {
            'Authorization': response['result']['token_type']+' '+response['result']['access_token'],
        }

        # start = '1646105850000' #2022-03-01
        # start = '1648864830000' #2022-04-02 10:00:00
        # end = '1648870230000'  #2022-04-02  11:30:00

        start = '1648893600000' #2022-04-02 18:00:00
        end = '1648898037000'  #2022-04-02  19:13:56
        # end = '1648893660000'  #2022-04-02  18:01:00

        # start = (int(time.mktime(time.strptime(pdata.start_time, "%Y-%m-%d %H:%M:%S"))) + 8*3600) * 1000
        # end = (int(time.mktime(time.strptime(pdata.end_time, "%Y-%m-%d %H:%M:%S"))) + 8*3600) * 1000

        # print('start:',start)
        # print('end:',end)

        
        trade = url + 'private/get_user_trades_by_instrument_and_time?count=1&end_timestamp='+str(end)+'&instrument_name=BTC-PERPETUAL&start_timestamp='+str(start)+'&sorting=desc&include_old=true'
        response = requests.get(trade,proxies=proxy,timeout=15,headers=headers)

        # response = requests.get('https://www.youtube.com',proxies=proxy,timeout=15)
        print(response.text)
        return




        # 找到时间范围的起始点seq
        # 之后1000累积
        start_seq = json.loads(response.text)['result']['trades'][0]['trade_seq']

        allTrade = []

        start_seq = int(start_seq)
        end_seq =  start_seq+999

        index = 1

        print(start_seq, end_seq) 
        # 67737102 67738101
        
        while True:

            trade = url + 'private/get_user_trades_by_instrument?count=1000&end_seq='+ str(end_seq)+'&instrument_name=BTC-PERPETUAL&include_old=true&sorting=asc&start_seq='+ str(start_seq)
            response = requests.get(trade,proxies=proxy,timeout=60,headers=headers)


            # print(start_seq, end_seq)

            response = json.loads(response.text)['result']['trades']

            # print(len(response))

            for item  in response:
                allTrade.append(item)

            # index+=1
            # if index>1:
            #     break
            # 1648895436326
            # 1648898036000


            if int(allTrade[len(allTrade)-1]['timestamp']) > int(end):
                print('超出',allTrade[len(allTrade)-1]['timestamp'])
                break
            else:
                start_seq = int(end_seq)+1
                end_seq =  start_seq+999

        temp = []
        for item in allTrade:
            item['Date'] = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(item['timestamp']/1000))

            if item['direction'] == 'buy':
                item['Side'] = 'open buy'
            else:    
                item['Side'] = 'close sell'

            #item['Equity'] = item['equity']
            
            item['Price'] = item['price']
            item['Amount'] = item['amount']
            item['Instrument'] = item['instrument_name']
            
            if item['timestamp'] < int(end):
                temp.append(item)

        allTrade = temp

        return allTrade


if __name__ =="__main__":
    get_trade_record()