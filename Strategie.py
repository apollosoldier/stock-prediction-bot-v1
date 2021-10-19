# -*- coding: utf-8 -*-

import os, numpy
class Strategie:    
    def __init__(self, parameters, portfolio_value=1000):
        self.parameters              = parameters
        self.portfolio_value         = portfolio_value
        self.portfolio_Initial_value = portfolio_value        
        self.fees                    = 0.10 / 100

        self.max_buy_number          = 4
        self.buy_iteration           = 0        
        self.free_margin             = (1 - 2*self.fees) * portfolio_value
        self.buy_unit                = self.free_margin/self.max_buy_number

        
        self.total_gain              = 0
        self.IS_BUYING_MODE          = True
        self.Buying_price            = 0
        self.Average_Buying_Price    = 0
        self.Buying_price_stored     = []
        self.Selling_price           = 1e10
        self.Counter                 = 0
        self.profit_loss             = 0


        self.take_profit             = parameters['take_profit'] /100
        self.stop_loss               = parameters['stop_loss']   /100
        self.Buy_history             = {'Price': [], 'Position': []}
        self.Sell_history            = {'Price': [], 'Position': []}

        self.Time                    = None
        self.Price                   = 0
        self.Volume                  = 0

        
        self.buy_or_sell             = None

        self.dirname                 = os.path.join("log", self.parameters['pair'])

        if not os.path.isdir('log'):
            os.mkdir('log')
        if not os.path.isdir(self.dirname):
            os.mkdir(self.dirname)                
        self.filebasename            = "Pair_%s_Interval_%d_smothingH%s_SmothingH%s" %(self.parameters['pair'],
                                                                                       self.parameters['interval'],
                                                                                       self.parameters['smothingH168'],
                                                                                       self.parameters['smothingH24'])
        
        self.logfile                 = os.path.join(self.dirname,self.filebasename + '.log')
        self.Reset_MIN_MAX_Value()        
        if os.path.isfile(self.logfile):
            os.remove(self.logfile)

        msg4  = "=====================================================================\n"
        msg4 += "Pair         : %s   \n" %(self.parameters['pair'])
        msg4 += "Interval     : %d   \n" %(self.parameters['interval'])
        msg4 += "smothingH4   : MA%s \n" %(self.parameters['smothingH4'])
        msg4 += "smothingH24  : MA%s \n" %(self.parameters['smothingH24'])
        msg4 += "smothingH168 : MA%s \n" %(self.parameters['smothingH168'])        
        msg4 += "Take Profit  : %s   \n" %(self.parameters['take_profit'])
        msg4 += "Stop Loss    : %s   \n" %(self.parameters['stop_loss'])
        msg4 += "Exp Mode     : %s   \n" %(self.parameters['exp_mode'])
        msg4 += "Localtime    : %s   \n" %(self.parameters['localtime'])
        msg4 += "=====================================================================\n"
        with open(self.logfile, 'w') as fw:
            fw.write(msg4)
    
    def Reset_MIN_Value(self):
        self.Minus_value       = {}
        self.Buying_Threshold  = {}
        for idx in ['1H', '4H', '24H', '168H']:
            self.Minus_value[idx]     = 1e10
            self.Buying_Threshold[idx]= 1e10 
    def Reset_MAX_Value(self):
        self.Maxus_value       = {}
        self.Selling_Threshold = {}
        for idx in ['1H', '4H', '24H', '168H']:
            self.Maxus_value[idx]      = -1e10
            self.Selling_Threshold[idx]= -1e10 
            
    def Reset_MIN_MAX_Value(self):
        self.Reset_MAX_Value()
        self.Reset_MIN_Value()
    
    def Set_MIN_MAX_Val(self):
        
        for idx in ['1H', '4H', '24H', '168H']:
            if self.Maxus_value[idx] < self.MeanPrice[idx]:
                 self.Maxus_value[idx]       = self.MeanPrice[idx]
                 self.Selling_Threshold[idx] = (1-self.stop_loss) * self.Maxus_value[idx]
                 # print("New MeanPrice%s Maximum value     : %0.5f"   %(idx,self.Maxus_value[idx]))
                 # print("New MeanPrice%s Selling Threshold : %0.5f\n" %(idx,self.Selling_Threshold[idx]))
            
            if self.Minus_value[idx] > self.MeanPrice[idx]:
                 self.Minus_value[idx]      = self.MeanPrice[idx]
                 self.Buying_Threshold[idx] = 1.01 * self.Minus_value[idx]
                 
                 # print("New MeanPrice%s Minimum value    : %0.5f"   %(idx,self.Minus_value[idx]))
                 # print("New MeanPrice%s Buying Threshold : %0.5f\n" %(idx,self.Buying_Threshold[idx]))
                                                            
    def save_file(self, MODE, msg=''):
        if MODE == 'BUY':
            msg4  = "BUY             :: Time           : %s   \n" % self.Time
            msg4 += "                :: Pair           : %s   \n" % self.parameters['pair']
            msg4 += "                :: Counter        : %d   \n" % self.Counter
            msg4 += "                :: Buying price   : %0.5f\n" % self.Buying_price            
            msg4 += "                :: Av.Buying price: %0.5f\n" % self.Average_Buying_Price
            msg4 += "                :: Volume         : %0.5f\n" % self.Volume
            
            msg4 += "                :: MeanPrice H4   : %0.5f\n" % self.MeanPrice['4H']
            msg4 += "                :: MeanPrice H24  : %0.5f\n" % self.MeanPrice['24H']
            msg4 += "                :: MeanPrice H168 : %0.5f\n" % self.MeanPrice['168H']

            msg4 += "                :: RSI       H1   : %0.5f\n" % self.RSI['1H']
            msg4 += "                :: RSI       H4   : %0.5f\n" % self.RSI['4H']
            msg4 += "                :: RSI       H24  : %0.5f\n" % self.RSI['24H']
            msg4 += "                :: RSI       H168 : %0.5f\n" % self.RSI['168H']

            msg4 += "                :: AO_H4_H24      : %0.5f\n" % self.AO_H4_H24            
            msg4 += "                :: AO             : %0.5f\n" % self.AO
         
        elif MODE == 'SELL':                
            gain                     = (self.Selling_price-self.Average_Buying_Price) * self.Volume
            fees                     = self.fees * self.Volume * self.Average_Buying_Price
            self.Volume              = 0 
            self.Buying_price        = 0
            self.Buying_price_stored = []
            self.portfolio_value    += gain - fees            
            self.profit_loss         = (self.portfolio_value-self.portfolio_Initial_value)

            msg4  = "%s :: Time         : %s   \n" % (msg,self.Time)
            msg4 += "                :: Pair           : %s   \n" % (self.parameters['pair'])
            msg4 += "                :: Counter        : %d   \n" % (self.Counter)
            msg4 += "                :: Volume         : %0.5f\n" % (self.Volume)
            msg4 += "                :: Selling price  : %0.5f\n" % (self.Selling_price)
            msg4 += "                :: MeanPrice H4   : %0.5f\n" % self.MeanPrice['4H']
            msg4 += "                :: MeanPrice H24  : %0.5f\n" % self.MeanPrice['24H']
            msg4 += "                :: MeanPrice H168 : %0.5f\n" % self.MeanPrice['168H']

            msg4 += "                :: RSI       H1   : %0.5f\n" % self.RSI['1H']
            msg4 += "                :: RSI       H4   : %0.5f\n" % self.RSI['4H']
            msg4 += "                :: RSI       H24  : %0.5f\n" % self.RSI['24H']
            msg4 += "                :: RSI       H168 : %0.5f\n" % self.RSI['168H']

            msg4 += "                :: AO_H4_H24      : %0.5f\n" % self.AO_H4_H24            
            msg4 += "                :: AO             : %0.5f\n" % self.AO
            msg4 += "                :: Profit/Loss    : %0.5f %s\n" % (gain,'EUR')
            msg4 += "                :: Porfolio       : %0.5f\n"    % (self.portfolio_value)
            msg4 += "                :: Total (P/L)    : %0.5f %s\n" % (self.profit_loss,'EUR')

        print(msg4)
        with open(self.logfile, 'a') as fw:
            fw.write(msg4 + '\n')
                        
    def MinMax(self, Data):
        self.Time                  = Data['Time']
        self.Price                 = Data['MeanPrice']['1H']
        self.MeanPrice             = Data['MeanPrice']
        self.RSI                   = Data['RSI']
        self.AO                    = Data["AO"]
        self.delta_ao              = Data["Delta_AO"]                
        self.Counter               = Data['Counter']
        self.AO_H4_H24             = self.MeanPrice['4H'] - self.MeanPrice['24H']

        if not self.parameters['exp_mode']:
            if Data["Counter"]< 1000:
                return
        elif self.parameters['smothingH168'] >  Data["Counter"]:
            return 

        # -------------------------------------------------------------
        # Trading strategy
        # -------------------------------------------------------------
        self.Set_MIN_MAX_Val()
        if self.buy_iteration < self.max_buy_number:                   
            if (sum(numpy.array(list(self.RSI.values())) <= 30) == 4 ) and self.MeanPrice['4H'] <= Data['Indicator']:
                if len(self.Buying_price_stored)>0 and self.Average_Buying_Price < self.Price:
                    return
                
                self.Volume                                 += self.buy_unit / self.Price
                fees                                         = self.fees * self.buy_unit
                self.portfolio_value                        -= fees
                self.Buying_price_stored.append(self.Price  + fees/self.Volume)
                self.Buying_price                            = self.Price
                self.Average_Buying_Price                    = numpy.mean(self.Buying_price_stored)
                self.buy_or_sell                             = 'buy'
                self.buy_iteration                          += 1

                self.Buy_history['Price'].append(self.Price)
                self.Buy_history['Position'].append(self.Counter)
                self.save_file("BUY")                
                self.Reset_MAX_Value()                
                print("\n")                    

        if self.Volume > 0:            
            if (sum(numpy.array(list(self.RSI.values())) >= 70) == 4) and self.delta_ao < 0 :
                self.Selling_price    = self.Price
                self.buy_iteration    = 0
                self.buy_or_sell      = 'sell'
                self.Sell_history['Price'].append(self.Price)                    
                self.Sell_history['Position'].append(self.Counter)
                self.save_file("SELL","SELL(take profit)")
                self.Reset_MIN_Value()