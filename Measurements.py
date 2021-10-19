# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 00:42:01 2020

@author: TRAORE
"""
import numpy

DATAS_4_Fig         = {"AO"          : {"1H-168H" : [],"4H-168H" : [],"24H-168H": []},
                       "Delta_AO"    : {"1H-168H" : [],"4H-168H" : [],"24H-168H": []},
                       "RSI"         : {"1H" : [], "4H" : [], "24H": [], "168H": []},
                       "MeanPrice"   : {"1H" : [], "4H" : [], "24H": [], "168H": []},
                       "metric_median_low"  : [],
                       "metric_median_high" : []}

class RSI():
    def __init__(self, period):
        self._period         = period
        self.rsi_stored_data = {}
        self.rsi             = {}
            
    def set_rsi(self, name, Price):
        if name not in self.rsi:
            self.rsi_stored_data[name]   = numpy.nan * numpy.ones(self._period)            
        # RSI computation
        self.rsi_stored_data[name][0:-1] = self.rsi_stored_data[name][1:]
        self.rsi_stored_data[name][-1:]  = Price

        delta                      = numpy.diff(self.rsi_stored_data[name])
        dUp                        = delta[delta>0]
        dDown                      = abs(delta[delta<0])
        
        avg_gain                   = numpy.sum(dUp)                        
        avg_loss                   = numpy.sum(dDown)

        if avg_loss == 0 :
            if avg_gain == 0 :
                self.rsi[name]     = numpy.nan
            else:
                self.rsi[name]     = 100                                    
        else:
            rs                     = avg_gain / avg_loss
            self.rsi[name]         = 100 - (100 / (1 + rs))
            
    def get_rsi(self, name):
        return self.rsi[name]
    def get_rsi_values(self):
        return self.rsi
# --------------------------------- AO --------------------------------
class AO():
    def __init__(self, period):
        self._period        = period
        self.ao             = {}
        self.delta_ao       = {}
        self.ao_stored_data = {}
        
    def set_ao(self, name, value):
        if name not in self.ao.keys():
            self.ao_stored_data[name]  =  numpy.nan * numpy.ones(self._period)

        self.ao[name]                  = value
        self.ao_stored_data[name][:-1] = self.ao_stored_data[name][1:]
        self.ao_stored_data[name][-1]  = value
        self.delta_ao[name]            = numpy.mean(numpy.diff(self.ao_stored_data[name]))
        
    def get_ao(self,name):
        return self.ao[name]
    
    def get_delta_ao(self,name):
        return self.delta_ao[name]
    
    def get_ao_values(self):
        return self.ao, self.delta_ao