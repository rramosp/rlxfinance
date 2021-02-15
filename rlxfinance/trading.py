
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools, os
import bokeh.plotting as bplot
import matplotlib.dates as mdates
from progressbar import progressbar as pbar

class Trade:
    
    BUY_THEN_SELL = 1
    SELL_THEN_BUY = 0
    
    def __init__(self, date, buyorsell,  price_sell_to_market, price_buy_from_market, volume=1, **openinfo):
        self.open_date = date
        self.buyorsell = buyorsell 
        self.open_price_sell_to_market = price_sell_to_market
        self.open_price_buy_from_market = price_buy_from_market
        self.volume = volume
        self.isclosed = False
        self.openinfo = openinfo
        
    def close(self, date, price_sell_to_market, price_buy_from_market, **closeinfo):
        self.close_date = date
        self.close_price_sell_to_market = price_sell_to_market
        self.close_price_buy_from_market = price_buy_from_market
        self.isclosed = True
        self.closeinfo = closeinfo
        if self.buyorsell==self.BUY_THEN_SELL:
            self.pnl = (self.close_price_sell_to_market - self.open_price_buy_from_market)*self.volume
        else:
            self.pnl = (self.open_price_sell_to_market - self.close_price_buy_from_market)*self.volume
       
    def is_buy_then_sell(self):
        return self.buyorsell == self.BUY_THEN_SELL

    def is_sell_then_buy(self):
        return self.buyorsell == self.SELL_THEN_BUY
    
    def __repr__(self):
        s = "opendate           : " + str(self.open_date)
        if self.isclosed:
            s += "\nclose date         : " + str(self.close_date)
        if self.buyorsell==self.BUY_THEN_SELL:
            s += "\noperation          : BUY-THEN-SELL"
            s += "\nopen price (buy)   : %.4f"%self.open_price_buy_from_market
            s += "\nopen spread        : %.4f"%(self.open_price_buy_from_market - self.open_price_sell_to_market)
            if self.isclosed:
                s+= "\nclose price (sell) : %.4f"%self.close_price_sell_to_market
                s += "\nclose spread       : %.4f"%(self.close_price_buy_from_market - self.close_price_sell_to_market)
        else:
            s += "\noperation          : SELL-THEN-BUY"
            s += "\nopen price (sell)  : %.4f"%self.open_price_sell_to_market
            s += "\nopen spread        : %.4f"%(self.open_price_buy_from_market - self.open_price_sell_to_market)
            if self.isclosed:
                s += "\nclose price (buy)  : %.4f"%self.close_price_buy_from_market
                s += "\nclose spread       : %.4f"%(self.close_price_buy_from_market - self.close_price_sell_to_market)
        s += "\nvolume             : %.2f"%self.volume
        if self.isclosed:
            s += "\npnl                : %+.4f"%self.pnl
            
        s += "\nstatus             : " + ("closed" if self.isclosed else "open")
        s += "\nopen info          : "+str(self.openinfo)
        if self.isclosed:
            s += "\nclose info         : "+str(self.closeinfo)
        
        return s
    
    def as_series(self):
        r = {"open_date": self.open_date}
        r = {**r, "close date": self.close_date if self.isclosed else None,
                  "volume": self.volume, 
                  "status": ("closed" if self.isclosed else "open"), 
                  "pnl": self.pnl if self.isclosed else None,
                  **self.openinfo,
                  **(self.closeinfo if self.isclosed else {}) }
            
        if self.buyorsell==self.BUY_THEN_SELL:
            r = {**r, "operation":    "BUY-THEN-SELL", 
                      "open_price":   self.open_price_buy_from_market, 
                      "open_spread":  self.open_price_buy_from_market - self.open_price_sell_to_market,
                      "close_price":  self.close_price_sell_to_market if self.isclosed else None,
                      "close_spread": (self.close_price_buy_from_market - self.close_price_sell_to_market) if self.isclosed else None }
        else:
            r = {**r, "operation":    "SELL-THEN-BUY",
                      "open_price":   self.open_price_sell_to_market,
                      "open_spread":  (self.open_price_buy_from_market - self.open_price_sell_to_market),
                      "close_price":  self.close_price_buy_from_market if self.isclosed else None,
                      "close_spread": (self.close_price_buy_from_market - self.close_price_sell_to_market) if self.isclosed else None }

        
        return r
    
    def plot(self, kinetics, tdelta="5min", additional_kinetics = []):
        tk = kinetics[self.open_date-pd.Timedelta(tdelta):self.close_date+pd.Timedelta(tdelta)]
        fig = plt.figure(figsize=(20,3))
        plt.plot(tk.index, tk.price_buy_from_market, label="price buy from market", color="red")
        plt.plot(tk.index, tk.price_sell_to_market, label="price sell from market", color="blue")
        if self.is_sell_then_buy():
            plt.scatter(self.open_date, self.open_price_sell_to_market, color="blue", label="OPEN SELL")
            plt.scatter(self.close_date, self.close_price_buy_from_market, color="red", label="CLOSE BUY")
        else:
            plt.scatter(self.open_date, self.open_price_buy_from_market, color="red", label="OPEN BUY")
            plt.scatter(self.close_date, self.close_price_sell_to_market, color="blue", label="CLOSE SELL")

        plt.plot(tk.index, tk.smoothed, label="smoothed", color="black", alpha=.5)

        for i,k in enumerate(additional_kinetics):
            ak = k[self.open_date-pd.Timedelta(tdelta):self.close_date+pd.Timedelta(tdelta)]
            plt.plot(ak.index, ak.smoothed, color="orange", alpha=.5)


        plt.axvline(self.open_date, color="black", ls="--")
        plt.axvline(self.closeinfo["best_price_date"], color="black", ls="--", alpha=.5, label="ref to trailing stop")
        plt.axhline(self.closeinfo["best_price"], color="black", ls="--", alpha=.5)
        plt.grid();
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=6)

        fig = plt.figure(figsize=(20,1))
        plt.plot(tk.index, tk.vel, label="velocity", color="black", lw=3)
        for i,k in enumerate(additional_kinetics):
            ak = k[self.open_date-pd.Timedelta(tdelta):self.close_date+pd.Timedelta(tdelta)]
            plt.plot(ak.index, ak.vel, color="orange", alpha=.5)
        plt.grid();
        plt.legend();
        plt.title("velocity")
        plt.axvline(self.open_date, color="black", ls="--")    
        


class Trader:
    
    def __init__(self, absvel_min, trailing_stop, spread_range=[-np.inf, np.inf], price_std_range=[-np.inf, np.inf]):
        self.absvel_min = absvel_min
        self.trailing_stop = trailing_stop
        self.spread_range = spread_range
        self.price_std_range = price_std_range        
        assert price_std_range[0]<price_std_range[1], "invalid interval price std range, first must be smaller than second"
        assert spread_range[0]<spread_range[1], "invalid interval spread range, first must be smaller than second"
        self.kinetics_sets = {}

    def set_source_signal(self, wd, smoothed_column, rolling_std_period, smoothed_velocity_column=None, smoothed_acceleration_column=None):
        """
        wd : pd.DataFrame
            a time indexed dataframe with the following columns 'price','price_sell_to_market', 'price_buy_from_market'
        
        smoothed_column : str
            must exist in `wd` and will be the one that will be monitored for velocity, etc.
            
        rolling_std_period : str or pd.Timedelta
            period of time to computing rolling stdev of `price` column
        
        """
        
        
        required_columns = ['price','price_sell_to_market', 'price_buy_from_market']
        
        self.smoothed_velocity_column = smoothed_velocity_column
        self.smoothed_acceleration_column = smoothed_acceleration_column
        self.rolling_std_period = rolling_std_period

        if not np.alltrue([i in wd.columns for i in required_columns]):
            raise ValueError("missing columns, expecting "+str(required_columns))
        
        kinetics = wd[['price', 'price_sell_to_market', 'price_buy_from_market', smoothed_column]].copy()
        kinetics.columns = ['price', 'price_sell_to_market', 'price_buy_from_market', 'smoothed']
        
        if smoothed_velocity_column is None:
            kinetics['vel'] = kinetics.smoothed.diff()
        else:
            kinetics['vel'] = wd[smoothed_velocity_column].copy()

        if smoothed_acceleration_column is None:
            kinetics['acc'] = kinetics.smoothed.diff().diff()
        else:
            kinetics['acc'] = wd[smoothed_acceleration_column].copy()
            
        kinetics = kinetics.join(wd['price'].rolling(rolling_std_period).std().dropna().rename('price_std')).dropna()
        kinetics['spread'] = kinetics.price_buy_from_market - kinetics.price_sell_to_market
        
        self.kinetics = kinetics

        self.current_kinetics_key = "%s__%s__%s"%(smoothed_column, str(smoothed_velocity_column), str(smoothed_acceleration_column))
        self.kinetics_sets[self.current_kinetics_key] = kinetics


        # use velocity computed from kalman smoothed
        """
        kinetics = ts.signal_kinetics(wd[monitored_column], dt="1min").dropna()
        kinetics = kinetics.join(wd['price'].rolling(rolling_std_period).std().dropna().rename('price_std'))
        kinetics = kinetics.join(wd[['price', 'price_sell_to_market', 'price_buy_from_market']])

        """

        
    def show_kinetics_sample(self, tdelta="2h"):
        i = self.kinetics.index[np.random.randint(len(self.kinetics))]
        k = self.kinetics[i:i+pd.Timedelta(tdelta)]
        plt.figure(figsize=(20,3))
        plt.plot(k.index, k.price, color="black", label="price", ls="--")
        plt.plot(k.index, k.price_buy_from_market, color="red", label="price buy from market", alpha=.5)
        plt.plot(k.index, k.price_sell_to_market, color="blue", label="price sell to market", alpha=.5)
        plt.plot(k.index, k.smoothed, color="black", label="smoothed", alpha=.5)

        for key,kin in self.kinetics_sets.items():
            if key==self.current_kinetics_key:
                continue
            ak = kin[i:i+pd.Timedelta(tdelta)]
            plt.plot(ak.index, ak.smoothed, color="orange", alpha=.5)

        plt.grid()
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4)
        
        plt.figure(figsize=(20,1))
        plt.plot(k.index, k.vel, color="black")
        plt.title("velocity")
        for key,kin in self.kinetics_sets.items():
            if key==self.current_kinetics_key:
                continue
            ak = kin[i:i+pd.Timedelta(tdelta)]
            plt.plot(ak.index, ak.vel, color="orange", alpha=.5)
        plt.grid()

        plt.figure(figsize=(20,1))
        plt.plot(k.index, k.acc, color="black")
        for key,kin in self.kinetics_sets.items():
            if key==self.current_kinetics_key:
                continue
            ak = kin[i:i+pd.Timedelta(tdelta)]
            plt.plot(ak.index, ak.acc, color="orange", alpha=.5)
        plt.grid()
        plt.title("acceleration")


    def show_kinetics_distributions(self):
        plt.figure(figsize=(25,3))
        plt.subplot(161)
        k = self.kinetics.vel.values
        k = k[pd.Series(k).between(*np.percentile(k, [1,99]))]
        plt.hist(k, bins=100, density=True);
        plt.title("kalman smoothed velocity")

        plt.subplot(162)
        k = self.kinetics.acc.values
        k = k[pd.Series(k).between(*np.percentile(k, [1,99]))]
        plt.hist(k, bins=100, density=True);
        plt.title("kalman smoothed acceleration")

        plt.subplot(163)
        k = self.kinetics.price_std.values
        k = k[pd.Series(k).between(*np.percentile(k, [1,99]))]
        plt.hist(k, bins=100, density=True);
        plt.title("price std")

        plt.subplot(164)
        k = self.kinetics.price.diff().dropna().values
        k = k[pd.Series(k).between(*np.percentile(k, [1,99]))]
        plt.hist(k, bins=100, density=True);
        plt.title("price velocity")

        plt.subplot(165)
        k = self.kinetics.price.diff().diff().dropna().values
        k = k[pd.Series(k).between(*np.percentile(k, [1,99]))]
        plt.hist(k, bins=100, density=True);
        plt.title("price acceleration")

        plt.subplot(166)
        k = self.kinetics.spread
        k = k[pd.Series(k).between(*np.percentile(k, [1,99]))]
        plt.hist(k, bins=100, density=True);
        plt.title("spread")      
        

    def plot_pnl(self):
        td = self.get_trades()
        plt.figure(figsize=(20,3))
        plt.plot(td.open_date, td.pnl.cumsum())
        plt.grid();
        plt.axhline(0, color="black")
        plt.ylabel("accumulated PNL")




    def run(self):
    
        kinetics = self.kinetics
    
        TRADING, MONITORING = 0, 1
        
        state = MONITORING
        
        current_trade = None
        self.trades = []
        
        for date, i in pbar(kinetics.iterrows(), max_value=len(kinetics)):
            if state == MONITORING:
                if np.abs(i.vel)>self.absvel_min and \
                   i.spread >= self.spread_range[0] and i.spread <= self.spread_range[1] and \
                   i.price_std >= self.price_std_range[0] and i.price_std <= self.price_std_range[1]:
                    
                    state = TRADING                    
                    
                    open_info = {"velocity": np.round(i.vel,4), "acceleration": np.round(i.acc, 4), "price_std": np.round(i.price_std,6)}
                    
                    if i.vel>0:                        
                        # if price rising, buy then sell
                        current_trade = Trade(date, Trade.BUY_THEN_SELL, i.price_sell_to_market, i.price_buy_from_market, **open_info)     
                        best_price = i.price
                        best_price_date = date
                    else:
                        # otherwise sell then buy
                        current_trade = Trade(date, Trade.SELL_THEN_BUY, i.price_sell_to_market, i.price_buy_from_market, **open_info)
                        best_price = i.price
                        best_price_date = date
            else:
                
                # if waiting to sell, current sell price must not fall beyond trailing_stop from best price
                # if waiting to buy, current buy price must not raise over trailing_stop from best price                
                delta_best = best_price - i.price
                if ( current_trade.is_buy_then_sell() and ( delta_best > self.trailing_stop) ) or \
                   ( current_trade.is_sell_then_buy() and (-delta_best > self.trailing_stop) ) :                    
                    
                    current_trade.close(date, i.price_sell_to_market, i.price_buy_from_market, best_price=best_price, best_price_date=best_price_date)
                    state = MONITORING
                    self.trades.append(current_trade)
                
                else:
                        
                    # update best price if that's the case
                    if current_trade.is_buy_then_sell() and i.price > best_price:
                        best_price = i.price
                        best_price_date = date
                        
                    if current_trade.is_sell_then_buy() and i.price < best_price:
                        best_price = i.price
                        best_price_date = date
                
                
    def get_trades(self):
        return pd.DataFrame([i.as_series() for i in self.trades])



