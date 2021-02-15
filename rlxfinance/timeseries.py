import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from os import listdir
from os.path import isfile, join

from progressbar import progressbar as pbar
from progressbar import ProgressBar
from scipy.optimize import minimize

from numpy.random import randn
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import JulierSigmaPoints
import bokeh.plotting as bplot

def get_price_for_day(day, prices_dir='data/signals/market_price/all', sample_freq=None):
    df = pd.read_hdf(prices_dir+'/'+day+'.hd5')
    df.index = [pd.Timestamp(i, unit='ms') for i in df.index]
    df.columns=["price"]
    if sample_freq is not None:
        df = df.resample(sample_freq).first()
    df = df[df.price!=0].copy()
    return df

def get_price_series(day, n_days=1, prices_dir='data/signals/market_price/all', sample_freq=None):
    """
    assumes a set of hdf files, one per day, with one column "price", and timedate index as unixtime
    """

    r = []
    i = 0
    pbar = ProgressBar(max_value=n_days)
    while len(r)<n_days:
        date = (pd.Timestamp(day)+pd.Timedelta("%dd"%i)).strftime("%Y-%m-%d")
        i+=1
        try:
            dp = get_price_for_day(date, prices_dir, sample_freq=sample_freq)
            r.append([date, dp])
        except:
            pass
        pbar.update(len(r))
    pall = pd.concat([i[1] for i in r])
    return pall

class KineticUKF:
    
    def __init__(self, c1=1, Qvar=1e-5, dtol=1e-5):
        """
        Kinetic Unscented Kalman Filter

        defined with the following kinetic model (x: pos, v: velocity, a: acceleration)
        
        x[t+1] = x[t] + v[t]*dt + a[t]*dt**2
        v[t+1] = v[t] + c1*a[t]*dt
        a[t+1] = a[t]
        
        
        Parameters:
        -----------
        
        c1 : float
            constant c1 in the kinetic equations
             
        Qvar : float
            variance for initial white noise Q matrix
             
        dtol : float
            tolerance to detect stability and convergence (lower is stricter)
        """
        
        self.c1 = c1
        self.Qvar = Qvar
        self.dtol = dtol
        
    def run(self, signal):
        
        def zeronormalize(x):
            """
            normalizes to [-1,1] keeping zero as the midpoint and preserving NaN's
            """
            xc = x.copy()
            xc[np.isnan(xc)] = 0
            r = np.zeros(xc.shape)
            
            gt0 = xc>0
            maxgt0 = np.max(xc[gt0])
            lt0 = xc<0
            minlt0 = np.min(xc[lt0])
            
            k = np.max([np.abs(minlt0), np.abs(maxgt0)])
            
            r[gt0] = xc[gt0]/k
            r[lt0] = xc[lt0]/k

            r[np.isnan(x)]=np.nan
            return r        
        
        def fx(x, dt):
            xout = np.empty_like(x)
            xout[0] = x[1] * dt + x[0] + x[2]*dt**2
            xout[1] = x[1] + self.c1*x[2]*dt
            xout[2] = x[2]
            return xout

        def hx(x):
            return x[:1] # return position [x] 
    
        sigmas = JulierSigmaPoints(n=3, kappa=1)        
        
        ukf = UnscentedKalmanFilter(dim_x=3, dim_z=1, dt=1., hx=hx, fx=fx, points=sigmas)
        #ukf.P *= 10
        #ukf.R *= 1.5
        ukf.Q = Q_discrete_white_noise(3, dt=1., var=self.Qvar)

        zs, xs, vs, ac, ps = [], [], [], [], []
        for i in range(len(signal)):
            z = signal[i]
            ukf.predict()
            ukf.update(z)
            xs.append(ukf.x[0])
            vs.append(ukf.x[1])
            ac.append(ukf.x[2])
            zs.append(z)
            ps.append(ukf.P)
        zs = np.r_[zs]
        xs = np.r_[xs]
        vs = np.r_[vs]
        ac = np.r_[ac]
        ps = np.r_[ps]
        # converged if the diagonal of covariance matrix P does not change in the second half of the signal
        converged = np.alltrue(np.r_[[np.std(ps[-len(ps)//2:,i,i]) for i in range(ps.shape[1])]]<self.dtol)

        # the latest points where convergence started
        try:
            converged_at = [np.argwhere(np.abs((ps[1:,i,i]-ps[:-1,i,i]))<self.dtol**2)[0][0] for i in range(ps.shape[1])]
        except IndexError as e:
            raise ValueError("filter did not converge at dtol %f"%self.dtol) from e
        converged_at = np.max(converged_at)

        # trim signals only after convergence
        xs[:converged_at] = np.nan
        zs[:converged_at] = np.nan
        vs[:converged_at] = np.nan
        ac[:converged_at] = np.nan
        
        vs = zeronormalize(vs)
        ac = zeronormalize(ac)        
        
        self.signal = signal  # signal
        self.xf = xs   # filtered signal
        self.vf = vs   # filter velocity
        self.af = ac   # filter acceleration
        
        self.converged = converged
        self.converged_at = converged_at
        return self
        
    def plot(self, window_size=None):
        
        # create a new plot with a title and axis labels
        fig = bplot.figure(title="kalman filter", x_axis_label='time', width=1500, height=400, y_axis_label='price')

        
        if window_size is None:
            window_size, _  = self.get_best_centered_mavg()
            best_label = " (best window size) " 
        else:
            best_label = ""
            
        csroll = pd.Series(self.signal).rolling(window=window_size, center=True).mean().values        
        fig.line(range(len(csroll)), csroll, legend_label="REF: centered mavg wsize = %d%s"%(window_size, best_label), color="red", alpha=.5)
        
        mse = self.diffmse_to_centered_mavg(window_size)
        fig.line(range(len(self.xf)), self.xf, legend_label="filter, diffmse to REF = %.2f"%mse, line_width=1, color="blue")
        fig.line(range(len(self.signal)), self.signal, legend_label="signal", line_width=2, color="black")
        
        sroll = pd.Series(self.signal).rolling(window=window_size).mean().values        
        fig.line(range(len(sroll)), sroll, legend_label="shifted mavg wsize=%d, diffmse to REF = %.2f"%(window_size, self.diffmse(sroll, csroll)), color="orange", alpha=.5)
        fig.legend.location = "top_left"
        fig.legend.click_policy="hide"
        # show the results
        bplot.show(fig)

    def get_best_centered_mavg(self):
        mse ={w:self.diffmse_to_centered_mavg(w) for w in range(1,500,2)}
        bestw = list(mse.keys())[np.argmin(list(mse.values()))]
        return bestw, mse[bestw]
    
    def diffmse(self, x, y):
        x=x[1:]-x[:-1]
        y=y[1:]-y[:-1]
        valid = (~np.isnan(x))&(~np.isnan(y))
        x = x[valid]
        y = y[valid]
        #return np.sqrt(np.sum((x-y)**2)/len(x))
        return np.mean(np.sign(x)==np.sign(y))

    def diffmse_to_centered_mavg(self, window_size):
        sroll = pd.Series(self.signal).rolling(window=window_size, center=True).mean().values
        return 1-self.diffmse(sroll, self.xf)


class KineticUKF_for_mavg:
    
    def __init__(self, target_wsize):
        """
        Finds the closest KineticUKF configuration to a given centered moving average
        """

        self.target_wsize = target_wsize   
    
    def fit(self, signal):
        
        self.last_cost = 100
        self.best_cost = np.inf
        self.best_params = [0,0]
        def cost(p):
            #c1,Qvar = np.exp(p)
            c1,Qvar = np.abs(p)
            print ("c1 %.10f"%c1, "Qvar %.10f"%Qvar, end=" ")
            try:
                r = KineticUKF(c1=c1, Qvar = Qvar).run(signal).diffmse_to_centered_mavg(window_size=self.target_wsize)
            except ValueError:
                r = self.last_cost
            print ("cost %.3f"%r, "best_cost %.3f"%self.best_cost)

            self.last_cost = r
            if r < self.best_cost:
                self.best_cost = r
                self.best_params = [c1, Qvar]
            return r       
        
        x0=(np.random.random(size=2)-.5)*.1
        print (x0)
        #x0[1] = x0[1]/1e3
        #x0=[1, 1e-2]

        cons = {'type':'ineq', 'fun': lambda x: x[0],
                'type':'ineq', 'fun': lambda x: x[1],
                'type':'ineq', 'fun': lambda x: 100-x[0],
                'type':'ineq', 'fun': lambda x: 100-x[1]
                  }

        ox = minimize(cost, x0=x0, method='SLSQP', constraints=cons)# , method="Nelder-Mead")#
        print (ox)
        if self.best_cost<ox.fun:
            c1, Qvar = self.best_params
        else:
            c1, Qvar = np.abs(ox.x)
        self.ku = KineticUKF(c1=c1, Qvar = Qvar).run(signal)
        return self.ku

class TimeSeriesConditionals:
    
    def __init__(self, df):
        self.df = df
        
    def rate_to_future(self, col_t_plus_dt, col_t, dt):
        df = self.df
        r = (df[col_t_plus_dt].shift(-dt)-df[col_t])/df[col_t]
        r = pd.DataFrame(r)
        if col_t==col_t_plus_dt:
            r.columns = ['%s/+%03d'%(col_t_plus_dt, dt)]    
        else:
            r.columns = ['%s+%03d/%s'%(col_t_plus_dt, dt, col_t)]    
        return r

    def rate_to_past(self, col_t_minus_dt, col_t, dt):
        df = self.df
        r = -(df[col_t_minus_dt].shift(dt)-df[col_t])/df[col_t]
        r = pd.DataFrame(r)
        if col_t_minus_dt==col_t:
            r.columns = ['%s/-%03d'%(col_t_minus_dt, dt)]
        else:
            r.columns = ['%s-%03d/%s'%(col_t, dt, col_t_minus_dt)]
        return r    
    
    def prob_TargetFuture_given_AlphaPast(self, col_target, col_alpha, dt_future, dt_past):
        df = self.df
        kt = self.rate_to_future(col_t_plus_dt=col_target, col_t=col_target, dt=dt_future)
        ka = self.rate_to_past(col_t_minus_dt=col_alpha, col_t=col_alpha, dt=dt_past)
        k = kt.join(ka).dropna().values
        return {"_|up":np.mean(k[:,1]>0), "up|_":np.mean(k[:,0]>0), 
                "_|dn":np.mean(k[:,1]<0), "dn|_":np.mean(k[:,0]<0), 
                "up|up": np.mean(k[k[:,1]>0][:,0]>0), "dn|dn": np.mean(k[k[:,1]<0][:,0]<0)}    
    
    def plot_probs_TargetFuture_given_AlphaPast(self, predict_spans, target_col, alpha_cols):
        
        rupup = np.zeros((len(predict_spans), len(alpha_cols)))
        rdndn = np.zeros((len(predict_spans), len(alpha_cols)))
        for pi,ps in enumerate(predict_spans):
            for wi,ws in enumerate(alpha_cols):
                prob = self.prob_TargetFuture_given_AlphaPast(target_col, ws, ps, 1)
                rupup[pi,wi] = prob["up|up"]
                rdndn[pi,wi] = prob["dn|dn"]        


        plt.figure(figsize=(14,4))
        plt.subplot(121)
        plt.contourf(rupup.T)
        plt.xticks(range(len(predict_spans)), predict_spans)
        plt.yticks(range(len(alpha_cols)), alpha_cols)
        plt.xlabel("predict span for %s"%target_col)
        plt.ylabel("alpha signal")
        plt.title("P(predict>0|alpha>0)")
        plt.colorbar()
        plt.subplot(122)
        plt.contourf(rdndn.T)
        plt.xticks(range(len(predict_spans)), predict_spans)
        plt.yticks(range(len(alpha_cols)), alpha_cols)
        plt.xlabel("predict span for %s"%target_col)
        plt.ylabel("alpha signal")
        plt.title("P(predict>0|alpha>0)")
        plt.colorbar()    


def signal_kinetics(signal, dt, dtol="1s"):
    """
    Computes the signal velocity and acceleration
    
    Parameters:
    ===========
    
    signal : pandas.Series
        A signal with time indices. 
        
    dt : str or pandas.Timedelta
        The size of the sliding window with which velocity and acceleration will be computed
        
    dtol: str or pandas.Timedelta
        Time "holes" in data might lead to data in the sliding window to contain a smaller
        time interval than dt. `dtol` controls the minimum size of the sliding window to
        compute valid velocity and acceleration. For instance, with dt="4min" and dtol="1min",
        rolling windows with only "3mins" of data will be ok, but smaller windows will yield nan.
        
    """

    dt = pd.Timedelta(dt)
    dtol = pd.Timedelta(dtol)
    onems = pd.Timedelta("1ms")
    onehr = pd.Timedelta("1h")
    def elapsed_time(x):
        xdt = x.index[-1]-x.index[0]
        xdv = x.iloc[-1]-x.iloc[0]

        if not dt-xdt<dtol+onems:
            return np.nan

        r = onehr.total_seconds()*xdv/xdt.total_seconds()
        return r

    velocity = signal.rolling(dt+onems).agg(elapsed_time)
    acceleration = velocity.diff(+1)
    kinetics = pd.DataFrame([velocity, acceleration], index=["vel", "acc"]).T
    return kinetics