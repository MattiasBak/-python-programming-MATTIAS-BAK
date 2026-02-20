import numpy as np
from scipy import stats

class LinearRegression:
    def __init__(self, fit_intercept=True, alpha =0.05):
        self._X=None
        self._y=None
        self.n=None
        self.d=None
        self.b= None
        self.fit_intercept = fit_intercept
        self.alpha= alpha
        self._fitted = False
        self._XtX_inv = None
        self.categorical_indx = None
        self.categories = None
        self.removed_rows = 0
        
    def _as_2d(self, X):
        X = np.asarray(X)
        if X.ndim ==1:  
            X = X.reshape (-1,1)
        return X
    
    
    def _as_1d(self, y):
        y = np.asarray(y,dtype=float)
        return y.reshape(-1)
    

    def _design(self,X):
        X = self._as_2d(X)
        if self.fit_intercept:
            ones = np.ones((X.shape[0],1), dtype=float)
            X = np.hstack((ones, X))
        return X
      
    def fit(self, X, y):

        X = self._as_2d(X)
        y = self._as_1d(y)
        mask = (~np.isnan(X).any(axis=1)) & (~np.isnan(y))
        X = X[mask]
        y = y[mask]
        Xd=self._prepare_X(X, fit=True)
        self.n = X.shape[0]
        self.p = Xd.shape[1]  
        self.d = self.p - int(self.fit_intercept) 
        self.removed_rows = np.sum(~mask)

        if X.shape[0] != y.shape[0]:
            raise ValueError('X and y must have the same number of rows')

        Xt = Xd.T
        XtX = Xt @ Xd
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(XtX)

        Xty = Xt @ y
        self._XtX_inv = XtX_inv
        self.b = XtX_inv @ Xty

        self._X = Xd
        self._y = y
        self._fitted = True

    def predict(self, X):
        if not self._fitted:
            raise RuntimeError("Model not fitted")

        Xd = self._prepare_X(X, fit=False)
        return Xd @ self.b


    def residuals (self):
        if not self._fitted:
            raise RuntimeError('This has not been fitted yet')
        yhat = self._X @self.b
        return self._y -  yhat
    

    def sse(self):
        r = self.residuals()
        return float (r.T @ r)
    

    def residual_variance (self):
        if not self._fitted:
            raise RuntimeError ('Unfortunatly you are not fitted so the calculation can not prosede.')
        return self.mse()


    def residual_std (self):
        if not self._fitted:
            raise RuntimeError ('This has not yet been fitted,')
        STD = (np.sqrt (self.residual_variance()))
        return float (STD)
    
    
    def rmse (self):
        if not self._fitted:
            raise RuntimeError ('This has not yet been fitted yet.')
        return float (np.sqrt(self.mse()))
    
    def sst (self):
         if not self._fitted:
             raise RuntimeError ('This has not been fitted yet')
         y_mean = self._y.mean()
         y_cof = self._y - y_mean
         return float (y_cof.T @ y_cof)


    def ssr (self):
        if not self._fitted:
            raise RuntimeError ('This has not been fitted yet')
        return float (self.sst() - self.sse())
    

    def r2(self):
        if not self._fitted:
            raise RuntimeError('This has not been fitted yet')
        SST = self.sst()
        if np.isclose (SST, 0.0):
            return float('nan')
        return float (1.0 - self.sse()/SST)
    
    def adjusted_r2(self):
        if not self._fitted:
            raise RuntimeError('Model not fitted')
        df= self.n - self.p
        if df <=0:
            return float('nan')
        return 1 - (1 - self.r2()) * (self.n - 1) / (df)


    def msr (self):
        if not self._fitted:
            raise RuntimeError ('This has not been fitted yet')
        if self.d <=0:
            return float ('nan')
        return float (self.ssr()/self.d)
    

    def mse (self):
        if not self._fitted:
            raise RuntimeError ('This has not been fitted yet')
        df_error = self.n - self.p
        if df_error <=0:
            raise ValueError('the degrees are insuficient for MSE')
        return float (self.sse()/df_error)


    def f_statistic (self):
        if not self._fitted:
            raise RuntimeError ('This has not yet been fitted')
        msr = self.msr()
        mse = self.mse()
        if not np.isfinite(msr) or not np.isfinite (mse) or mse == 0.0:
            return float('nan')        
        return float (msr/mse)
    

    def f_pvalue (self):
        if not self._fitted:
            raise RuntimeError('This has not yet been fitted')
        F = self.f_statistic()
        if not np.isfinite(F):
            return float('nan')
        df1 = self.d
        df2 = self.n - self.p
        if df1 <=0 or df2<=0:
            return float ('nan')
        return (stats.f.sf(F,df1,df2))
    

    def cov_matrix (self):
        if not self._fitted:
            raise RuntimeError ('This has not yet been fitted')        
        return self._XtX_inv * self.residual_variance()
    

    def standard_errors (self):
        if not self._fitted:
            raise RuntimeError ('This has not yet been fitted')
        C=self.cov_matrix()
        return np.sqrt(np.diag(C))
    

    def t_statistics (self):
        if not self._fitted:
            raise RuntimeError ('This has not yet been fitted')
        se=self.standard_errors()
        with np.errstate(divide='ignore', invalid='ignore'):
            return self.b/se
        

    def t_pvalues (self):
        if not self._fitted:
            raise RuntimeError('this has not yet been fitted')
        tvals = self.t_statistics()
        df = self.n - self.p
        if df <= 0:
            raise ValueError ('The degrees are of and cant be calculated')
        p = 2 * stats.t.sf(np.abs(tvals), df)
        return np.clip (p, 0.0, 1.0)
    

    def confidence_intervals (self, alpha = None):
        if not self._fitted:
            raise RuntimeError ('This has not yet been fitted')
        if alpha is None:
            alpha = self.alpha
        if not (0.0 < alpha < 1.0):
            raise ValueError ('Alpha must be in (0,1).')
        df = self.n - self.p
        
        if df <=0: raise ValueError ('Not enough degrees of freedom for confidence intervals')
        se = self.standard_errors()
        tcrit = stats.t.ppf (1.0 - alpha/2.0, df)
        lower = self.b - tcrit*se
        upper = self.b + tcrit*se
        return np.vstack ([lower, upper]).T
    
    @property
    def df_error(self):
        if not self._fitted:
            raise RuntimeError ('This has not yet been fitted')
        return self.n - self.p

    def pearson_X(self, X):
        Xn = self._prepare_X(X, fit=False)
        return self.pearson_matrix(Xn)

    def pearson_matrix(self, X):
        X = self._as_2d(X)
        d = X.shape[1]
        R = np.eye(d, dtype=float)
        P = np.full((d, d), np.nan, dtype=float)

        for i in range(d):
            for j in range(i + 1, d):
                xi = X[:, i]
                xj = X[:, j]
                mask = np.isfinite(xi) & np.isfinite(xj)
                xi = xi[mask]
                xj = xj[mask]
                if xi.size < 3 or xj.size < 3:
                    r = np.nan
                    p = np.nan
                else:
                    sxi = np.std(xi, ddof=1)
                    sxj = np.std(xj, ddof=1)

                    if np.isclose(sxi, 0.0) or np.isclose(sxj, 0.0):
                        r = np.nan
                        p = np.nan
                    else:
                        r, p = stats.pearsonr(xi, xj)

                R[i, j] = R[j, i] = r
                P[i, j] = P[j, i] = p

        return R, P
    

    def _prepare_X(self, X, fit=False):
        X = self._as_2d(X)

        cols = []

        if fit:
            self.categorical_indx = []
            self.categories = {}

        for i in range(X.shape[1]):
            col = X[:, i].reshape(-1,1)

    
            if col.dtype.kind in {'U','S','O'}:
                if fit:
                    self.categorical_indx.append(i)
                    self.categories[i] = np.unique(col)

                cats = self.categories[i]

                if not fit:
                    unknown = set(np.unique(col)) - set(cats)
                    if unknown:
                        raise ValueError(f"Unknown categories in column {i}: {unknown}")

                for c in cats[1:]:
                    cols.append((col == c).astype(float))

            else:
                cols.append(col.astype(float))

        X = np.hstack(cols)
        return self._design(X)
    
    def summary(self):
        if not self._fitted:
            raise RuntimeError("Model not fitted")

        names = ['Intercept'] + [f'x{i}' for i in range (len(self.b)-1)]
        se = self.standard_errors()
        t = self.t_statistics()
        p = self.t_pvalues()
        ci = self.confidence_intervals()

        return {
            'variables': names,
            "coefficients": self.b,
            "std_error": se,
            "t": t,
            "p": p,
            "CI_low": ci[:,0],
            "CI_high": ci[:,1],
            "R2": self.r2(),
            "Adj_R2": self.adjusted_r2(),
            "F": self.f_statistic(),
            "F_p": self.f_pvalue(),
            "RMSE": self.rmse()
        }






