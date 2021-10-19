# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 23:36:28 2020

@author: s90363
"""

import krakenex
import matplotlib.pyplot as plt
import argparse
import numpy
import os

from threading import Thread
from time import sleep, localtime
from datetime import datetime
from Strategie import Strategie
import Measurements

class Djagobot(Thread):
    def __init__(self, 
                 pair="XBTUSD", 
                 interval=1,
                 save_fig=False,
                 exp_mode=False,
                 loop=False,
                 key='key',
                 smothingH4=4,
                 smothingH24=24,
                 smothingH168=168,
                 take_profit=2.0,
                 stop_loss=2.0):
                
        Thread.__init__(self)        
        self.pair                    = pair
        self.interval                = interval
        self.save_fig                = save_fig
        self.exp_mode                = exp_mode
        self.loop                    = loop
        # self.smothingH4              = int(60*smothingH4/self.interval)
        # self.smothingH24             = int(60*smothingH24/self.interval)
        # self.smothingH168            = int(60*smothingH168/self.interval)        

        self.smothingH4              = smothingH4
        self.smothingH24             = smothingH24
        self.smothingH168            = smothingH168

        self.key                     = key
        self.Time_to_sleep           = 60
        self.last_time_ohlc          = None
        self.last_time_recent_trades = None
        self.Time                    = None
        self.Buying_price            = 0
        self.Selling_price           = 0
        self.Price                   = 0
        self._Is_running             = True
        self.IS_BUYING_MODE          = True
        # Measurements
        
        self._ao_computation         = Measurements.AO(3)
        self._rsi_computation        = Measurements.RSI(14)        
        # ------------
        self.take_profit             = take_profit
        self.stop_loss               = stop_loss        
        # Trading params
        self.order_params            = {"pair"     : self.pair, 
                                        "type"     : None,     # sell or buy
                                        "ordertype": "limit",
                                        "volume"   : None,
                                        "leverage" : 1}        
        # Strategie
        Time                         = localtime()
        self._localtime              = "%s_%s_%s_%sH_%sMin_%sSecs" % (Time[2], Time[1],Time[0], Time[3],Time[4], Time[5])        
        parameters                   = {'pair':pair,
                                        'interval':interval,
                                        #'metric':self.metric,
                                        'localtime':self._localtime,
                                        'exp_mode':self.exp_mode,
                                        'smothingH4':self.smothingH4,
                                        'smothingH24':self.smothingH24,
                                        'smothingH168':self.smothingH168,
                                        'take_profit': self.take_profit,
                                        'stop_loss': self.stop_loss}
        self.strategie               = Strategie(parameters)
        # Financial Indicators         
        self.AO                      = {}
        self.Delta_AO                = {}
        
        self.Metric                  = 0
        self._len_delta              = int(smothingH168/2)
        self.Counter                 = -self._len_delta

        # --------------------------
        self.Data_4_metrics          = numpy.nan * numpy.ones(smothingH168)
        self.Volume                  = numpy.nan * numpy.ones(smothingH168)

        # initialize Kraken API
        self.api                     = krakenex.API()
        self.api.load_key(self.key)    
        if self.save_fig:            
            self.DATAS               = Measurements.DATAS_4_Fig
            
    def get_result(self, Response):
        if not Response['error']:
            return Response['result']
        else:
            return None       

    def get_AssetPairs(self):
        return self.get_result(self.api.query_public('AssetPairs'))
                
    def get_ticker_information(self):
        params   = {"pair": self.pair}        
        Response = self.api.query_public('Ticker', data=params)
        Response = self.get_result(Response)        
        key      = list(Response)[0]  
        return Response[key]
        
    # Open-high-low-close chart
    def get_ohlc_data(self):           
        #print("Get OHLC data")
        params              = {"pair": self.pair, "interval": self.interval,  "since": self.last_time_ohlc}        
        Response            = self.api.query_public('OHLC', data=params)
        Response            = self.get_result(Response)
        try:            
            a ,  b              = Response
            if self.last_time_ohlc != Response[b]:                
                self.last_time_ohlc = Response[b]        
                return Response[a]
        except:
            return None
                           
    def get_recent_trades(self):        
        params                       = {"pair": self.pair, "since": self.last_time_recent_trades}
        Response                     = self.api.query_public('Trades', data=params)    
        Response                     = self.get_result(Response)
        a ,  b                       = Response
        self.last_time_recent_trades = Response[b]
        return Response[a]            

    def get_trade_balance(self):
        Response = self.api.query_private("TradeBalance")
        return self.get_result(Response) 
    
    def get_open_orders(self):
        Response = self.api.query_private("OpenOrders")
        return self.get_result(Response) 
    
    def get_closed_orders(self):
        Response = self.api.query_private("ClosedOrders")
        return self.get_result(Response) 
    
    def get_open_positions(self):
        Response = self.api.query_private("OpenPositions")
        return self.get_result(Response) 
    
    def add_standard_order(self):
        Response = self.api.query_private('AddOrder', data=self.order_params)           
        return self.get_result(Response) 

    
    def get_current_price(self):
        result = self.get_ticker_information()
        return float(result['c'][0])
            
    def update_ohlc_data(self):
        trades = self.get_ohlc_data()
        Data   = {}
        if trades:         
            for trade in trades:
                Time                     = float(trade[0])
                Open                     = float(trade[1])
                High                     = float(trade[2])
                Low                      = float(trade[3])
                Close                    = float(trade[4])                
                Vwap                     = float(trade[5])
                Volume                   = float(trade[6])
                Count                    = float(trade[7])
                self.Price               = Close                                
                # --------------------------
                # Time 
                # --------------------------
                self.Time                = datetime.fromtimestamp(Time).strftime("%A, %B %d, %Y %H:%M:%S")
                # --------------------------
                # Close 
                # --------------------------
                self.Data_4_metrics[0:-1]  = self.Data_4_metrics[1:]
                self.Data_4_metrics[-1:]   = Close
                self.Volume[0:-1]          = self.Volume[1:]
                self.Volume[-1:]           = Volume
                
                # Mean Price computation                
                self.MeanPriceH4            = sum(self.Volume[-self.smothingH4:]*self.Data_4_metrics[-self.smothingH4:])/sum(self.Volume[-self.smothingH4:])                
                self.MeanPriceH24           = sum(self.Volume[-self.smothingH24:]*self.Data_4_metrics[-self.smothingH24:])/sum(self.Volume[-self.smothingH24:])
                self.MeanPriceH168          = sum(self.Volume*self.Data_4_metrics)/sum(self.Volume)

                # self.MeanPriceH4            = numpy.mean(self.Data_4_metrics[-self.smothingH4:])
                # self.MeanPriceH24           = numpy.mean(self.Data_4_metrics[-self.smothingH24:])
                # self.MeanPriceH168          = numpy.mean(self.Data_4_metrics)
                
                # ------------------------- #
                #   Coputation and Storage  #
                # ------------------------- #                
                # ------------------------------ #
                # RSI computation                #
                # ------------------------------ #
                
                self._rsi_computation.set_rsi('1H',   self.Price)
                self._rsi_computation.set_rsi('4H',   self.MeanPriceH4)
                self._rsi_computation.set_rsi('24H',  self.MeanPriceH24)
                self._rsi_computation.set_rsi('168H', self.MeanPriceH168)                
                # ------------------------------ #
                # Awesome Oscillator computation #
                # ------------------------------ #
                # self.get_AO()
                self._ao_computation.set_ao('1H-168H',  self.Price        - self.MeanPriceH168)
                self._ao_computation.set_ao('4H-168H',  self.MeanPriceH4  - self.MeanPriceH168)
                self._ao_computation.set_ao('24H-168H', self.MeanPriceH24 - self.MeanPriceH168)

                #self.AO, self.Delta_AO = self._ao_computation.get_ao('4H-168H')
                
                # --------------------------------- #
                # Make data for strategie algorithm #
                # --------------------------------- #
                Data['Time']             = self.Time
                Data['Indicator']        = (Open+Close)/2
                Data['MeanPrice']        = {}
                Data['MeanPrice']['1H']  = self.Price                
                Data['MeanPrice']['4H']  = self.MeanPriceH4                
                Data['MeanPrice']['24H'] = self.MeanPriceH24
                Data['MeanPrice']['168H']= self.MeanPriceH168
                
                Data["Counter"]          = self.Counter 
                Data["RSI"]              = self._rsi_computation.get_rsi_values()
                Data["AO"]               = self._ao_computation.get_ao('4H-168H')
                Data["Delta_AO"]         = self._ao_computation.get_delta_ao('4H-168H')

                self.strategie.MinMax(Data)
                self.Counter             += 1                
                if self.save_fig:
                    if self.Counter >= 0:
                        # Saving for plotting                       
                        self.DATAS["RSI"]["1H"].append(self._rsi_computation.get_rsi('1H'))
                        self.DATAS["RSI"]["4H"].append(self._rsi_computation.get_rsi('4H'))
                        self.DATAS["RSI"]["24H"].append(self._rsi_computation.get_rsi('24H'))                        
                        self.DATAS["RSI"]["168H"].append(self._rsi_computation.get_rsi('168H'))

                        self.DATAS["AO"]["1H-168H"].append(self._ao_computation.get_ao('1H-168H'))
                        self.DATAS["AO"]["4H-168H"].append(self._ao_computation.get_ao('4H-168H'))
                        self.DATAS["AO"]["24H-168H"].append(self._ao_computation.get_ao('24H-168H'))

                        self.DATAS["Delta_AO"]["1H-168H"].append(self._ao_computation.get_delta_ao('1H-168H'))       
                        self.DATAS["Delta_AO"]["4H-168H"].append(self._ao_computation.get_delta_ao('4H-168H'))       
                        self.DATAS["Delta_AO"]["24H-168H"].append(self._ao_computation.get_delta_ao('24H-168H'))       

                        self.DATAS["MeanPrice"]["1H"].append(self.Price)
                        self.DATAS["MeanPrice"]["4H"].append(self.MeanPriceH4)
                        self.DATAS["MeanPrice"]["24H"].append(self.MeanPriceH24)
                        self.DATAS["MeanPrice"]["168H"].append(self.MeanPriceH168)
            return True
        else:
            return False            
    
    def run(self):
        print("Starting ...")
        if self.loop:        
            while self._Is_running:
                # try:            
                    if self.update_ohlc_data():
                        msg4  = "From                   : %s   \n" % self.pair
                        msg4  = "Time                   : %s   \n" % self.Time
                        msg4 += "MeanPrice H4           : %0.5f\n" % self.MeanPriceH4
                        msg4 += "MeanPrice H24          : %0.5f\n" % self.MeanPriceH24
                        msg4 += "MeanPrice H168         : %0.5f\n" % self.MeanPriceH168
                        msg4 += "Price                  : %0.5f\n" % self.Price                        
                        # msg4 += "AO                     : %0.5f\n" % self.AO
                        # msg4 += "RSI                    : %0.5f\n" % self.RSI
                        # msg4 += "Delta_AO(strategie)    : %0.5f\n" % self.strategie.delta_ao
                        # msg4 += "BUYING_MODE(strategie) : %s   \n" % self.strategie.IS_BUYING_MODE
                        print(msg4)
                    sleep(self.Time_to_sleep)
                # except KeyboardInterrupt :
                #     break
                # except Exception as e:
                #     print(e)
        else:
            self.update_ohlc_data()       
        print("End.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("KRAKEN BOT")
    parser.add_argument("-p",   "--pair",        type=str,   default = "ETHEUR",    help = "Pair")
    parser.add_argument("-k",   "--key",         type=str,   default = "key",       help = "Key Pair")
    parser.add_argument("-i",   "--interval",    type=int,   default = 60,          help = "Interval")
    parser.add_argument("-sh4", "--smothingH4",  type=int,   default = 4,           help = "smothing H4")
    parser.add_argument("-sh24","--smothingH24", type=int,   default = 24,          help = "smothing H24")
    parser.add_argument("-sh168","--smothingH168", type=int, default = 168,         help = "smothing H168")
    parser.add_argument("-tp",  "--take_profit", type=float, default = 5.0,         help = "Take Profit")
    parser.add_argument("-sl",  "--stop_loss",   type=float, default = 2.0,         help = "Stop Loss")

    parser.add_argument("-l", "--loop",           action="store_true",          help="Loop")
    parser.add_argument("-f", "--figure",         action="store_true",          help="Save/Plot figures")
    parser.add_argument("-e", "--exp",            action="store_true",          help="Experimentation mode")

    # parser.add_argument("-l", "--loop",           type=bool, default= False,         help="Loop")
    # parser.add_argument("-f", "--figure",         type=bool, default= True,         help="Save/Plot figures")
    # parser.add_argument("-e", "--exp",            type=bool, default= True,         help="Experimentation mode")
    args           = parser.parse_args()
    print(args)
    if not args.pair in ['all', 'All']:        
        info           = Djagobot(pair        = args.pair,
                                  interval    = args.interval,
                                  save_fig    = args.figure,
                                  exp_mode    = args.exp,
                                  loop        = args.loop,
                                  key         = args.key,
                                  smothingH4  = args.smothingH4,
                                  smothingH24 = args.smothingH24,
                                  smothingH168= args.smothingH168,
                                  take_profit = args.take_profit,
                                  stop_loss   = args.stop_loss)     
        info.run()
        if info.save_fig:
            plt.figure(1)
            plt.plot(info.DATAS["MeanPrice"]["1H"],    'b', label = 'Prices')
            plt.plot(info.DATAS["MeanPrice"]["4H"],  '--r', label = 'MeanPrice :: H4')
            plt.plot(info.DATAS["MeanPrice"]["24H"], '--g', label = 'MeanPrice :: H24')
            plt.plot(info.DATAS["MeanPrice"]["168H"],'--m', label = 'MeanPrice :: H168')

            plt.plot(info.strategie.Buy_history['Position'], info.strategie.Buy_history['Price'], 'go', label = 'Buy Price')
            plt.plot(info.strategie.Sell_history['Position'], info.strategie.Sell_history['Price'],'ro', label = 'Sell Price')            
            plt.legend()
            plt.grid()
            plt.ylabel('Prices')   
            plt.title("Prices - %s " %args.pair)
            fig_name   = os.path.join(info.strategie.dirname,"Figure_%s_%s.png" % (info.strategie.filebasename,"Price"))
            plt.savefig(fig_name, dpi = 1200)    
        
            plt.figure(2)
            plt.plot(info.DATAS["RSI"]["1H"],    'b', label = '1H   (14)')
            plt.plot(info.DATAS["RSI"]["4H"],  '--r', label = '4H   (14)')
            plt.plot(info.DATAS["RSI"]["24H"], '--g', label = '24H  (14)')            
            plt.plot(info.DATAS["RSI"]["168H"],'--m', label = '168H (14)')
            plt.legend()
            plt.grid()
            plt.title("RSI - %s " %args.pair)
            fig_name   = os.path.join(info.strategie.dirname,"Figure_%s_%s.png" % (info.strategie.filebasename,"RSI"))
            plt.savefig(fig_name, dpi = 1200)    
            
            plt.figure(3)
            plt.plot(info.DATAS["AO"]["1H-168H"],    'b', label = f'AO(1,168)')
            plt.plot(info.DATAS["AO"]["4H-168H"],  '--r', label = f'AO(4,168)')
            plt.plot(info.DATAS["AO"]["24H-168H"], '--g', label = f'AO(24,168)')
            plt.legend()
            plt.grid()            
            plt.title("AO - %s " %args.pair)
            fig_name   = os.path.join(info.strategie.dirname,"Figure_%s_%s.png" % (info.strategie.filebasename,"AO"))
            plt.savefig(fig_name, dpi = 1200)
            

            # plt.figure(4)
            # plt.plot(info.DATAS["metric_median_high"], 'b',    label = f'metric_median_high')
            # plt.legend()
            # plt.grid()            


            # plt.figure(5)
            # plt.plot(info.DATAS["metric_median_low"], 'b',    label = f'metric_median_low')
            # plt.legend()
            # plt.grid()            

            # plt.title("Delta AO  - %s" %args.pair)
            # fig_name   = os.path.join(info.strategie.dirname,"Figure_%s_%s.png" % (info.strategie.filebasename,"Delta_AO"))
            # plt.savefig(fig_name, dpi = 1200)
                        
    else:        
            assets         = Djagobot().get_AssetPairs()
            AssetPairs     = {}
            for name in assets:
                pair = assets[name]['altname']
                if ("EUR" in pair) and (".d" not in pair):
                    print("%s" % pair)
                    AssetPairs[pair] = Djagobot(pair        = pair,
                                                interval    = args.interval,
                                                save_fig    = args.figure,
                                                exp_mode    = args.exp,
                                                loop        = args.loop,
                                                key         = args.key,
                                                smothingH4  = args.smothingH4,
                                                smothingH24 = args.smothingH24,
                                                smothingH168= args.smothingH168,
                                                take_profit = args.take_profit,
                                                stop_loss   = args.stop_loss)     
                    AssetPairs[pair].run()   
                    sleep(1)
                    print("\n\n")
            # try:
            #     while True:
            #         pass
            # except :
            #     for name in assets:
            #         pair = assets[name]['altname']
            #         if "EUR" in pair:   
            #             AssetPairs[pair]._Is_running = False
            #             AssetPairs[pair].join()                    