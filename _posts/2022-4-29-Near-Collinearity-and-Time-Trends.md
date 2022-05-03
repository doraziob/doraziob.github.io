In this module, I review the problem of near-collinearity in the context of mean heterogeneity. I will briefly cover the problem, as well as the solution proposed in Michaelides and Spanos (2020). Python functions are defined for each step in the module.

## The Problem of Near-Collinearity in the Context of the Linear Regression Model

When considering the Linear Regression Model, near-collinearity can cause instability in estimates due to heightened sensitivity to small changes in the data. Notably, the problem starts when using first-order conditions to derive the OLS estimates:

### <center> $ {\hat{\boldsymbol{\beta}}} = (\mathbf{X}^\intercal\mathbf{X})^{-1}\mathbf{X}^\intercal\mathbf{y}$. <center>

Specifically, when the matrix $(\mathbf{X}^\intercal\mathbf{X})$ is ill-conditioned, the estimates ${\hat{\boldsymbol{\beta}}}$ may be unstable (Michaelides and Spanos, 2020). This problem often arises when fitting time polynomials in least squares estimation, where the time polynomials are used to capture different forms of mean heterogeneity. Michaelides and Spanos (2020) explores the issue of near-collinearity when using trend polynomials, and advises re-scaling the time variable and applying orthogonal polynomials when the number of trends is large (greater than 7).

## Scaling a Time Variable

The first step the authors propose to combat the problem of near-collinearity that stems from the inclusion of time polynomials is to scale the time variable via a monotonic transformation. Three, different transformations are presented, namely: (1) scale by the sample size, (2) use logarithmic transformations, or (3) scale the range to lie within the interval [-1,1]. In this example, the scaling is done using the transformation in Spanos (2019):

### <center> $ t_* = \frac{2t - n - 1}{n-1}, t = 1,\dots,n,$ <center>

where $n$ is the number of observations. I have defined a function, which creates the scaled time variable, below. 


```python
#import packages
import numpy as np
from numpy import size
import math
import sympy as sym
import pandas as pd
```


```python
def scaled_t(data):
    '''
    Parameters
    ----------
    data : pandas dataframe or array containing data

    Returns
    -------
    t_scaled : list containing scaled time variable
    '''
    
    n = size(data,0)
    t_scaled = []
    
    for t in range(1,n+1):
        t_s = (2*t - n - 1)/(n-1)
        t_scaled.append(t_s)
                   
    return t_scaled
```

### Example:


```python
#run the functions using example dataset

#generate dataset
data = np.random.normal(0,1,1000)

#create time variables
t = list(range(1,np.size(data)+1)) 
t_s = scaled_t(data)
```


```python
#create the dataframe containing the time variables and the data
df = pd.DataFrame()

df['t'] = t
df['t_s'] = t_s
df['x'] = data

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>t</th>
      <th>t_s</th>
      <th>x</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>-1.000000</td>
      <td>-1.624366</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>-0.997998</td>
      <td>-0.249450</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>-0.995996</td>
      <td>0.863277</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>-0.993994</td>
      <td>-0.022237</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>-0.991992</td>
      <td>0.978588</td>
    </tr>
  </tbody>
</table>
</div>



We now have a dataframe which contains the original time variable, the newly scaled time variable, and our data. 

## Orthogonal Polynomials

While the monotonic transformations offered above are found to reduce the effects $n$ has on conditioning, such transformations were found to not be enough to get rid of the problem of an ill-conditioned $(\mathbf{X}^\intercal\mathbf{X})$ altogether. Specifically, Michaelides and Spanos (2020) find that even with scaling, signs of an ill-conditioned $(\mathbf{X}^\intercal\mathbf{X})$ remained. 

The second step to render $(\mathbf{X}^\intercal\mathbf{X})$ well-conditioned is to replace the ordinary time polynomial with orthogonal polynomials. While there are many different types of orthogonal polynomials, both discrete and continuous, here I choose to use the discrete Chebyshev polynomial (of the first kind) as an example. The following equation is used to construct each polynomial term: 

### <center> $t_k(t_*) = \frac{(-1)^k2^kk!}{(2k)!}(1-t_*^2)^\frac{1}{2}(\frac{d}{dt_*})^k(1-t_*^2)^{k-\frac{1}{2}}$.


Here, the weight function here is given as 

### <center> $w(t_*) = (1-t_*^2)^{-\frac{1}{2}}$. 

Also, note that $t_* \in [-1,1]$. The Python function for constructing the Chebyshev polynomial is given below:


```python
#Chebyshev Polynomial of the First Kind
def Chebyshev_poly(t_s, trend): 
    '''
    Parameters
    ----------
    t_s: is a list containing the time variable
    trend: is an int describing the order of the trend (trend > 0)

    Returns
    -------
    Dictionary containing orthogonal trend polynomials up to order trend
    
    '''
    t = sym.symbols('t')
    
    t_poly = {}
    
    for k in range(1,trend+1):
        constant = ((-1)**k*2**k*math.factorial(k))/(math.factorial(2*k))
        w = (1 - t**2)**(0.5)
        der = (1 - t**2)**(k-0.5)
        ddt = sym.diff(der,t,k) 
        poly = sym.expand(constant*w*ddt)
        
        f = sym.lambdify(t,poly)
    
        t_poly['t_%d' %(k)] = []
        
        for i in range(0, size(t_s,0)):
            t_poly['t_%d' %(k)].append(f(t_s[i]))
            
    return t_poly
```

## Putting Everything Together

Ultimately, it was found in Michaelides and Spanos (2020) that using either one of these steps separately did not eliminate the problem of near-collinearity in the presence of higher-order trends. However, applying the rescaling and substituting the ordinary polynomial for the orthogonal polynomial (whose orthogonality is within the bounds [-1,1]) rendered $(\mathbf{X}^\intercal\mathbf{X})$ well-conditioned. Note that when the number of trend terms is low, say less than six, scaling alone is sufficient. The example below applies both the rescaling and generates the Chebyshev polynomial terms. 

### Example


```python
#run the functions using example dataset
trend = 3

#generate dataset
data = np.random.normal(0,1,1000)

#create time variables
t = list(range(1,np.size(data)))
t_s = scaled_t(data)
t_poly = Chebyshev_poly(t_s,trend)

#convert t_poly to a pandas dataframe
time_data = pd.DataFrame.from_dict(t_poly)
```

Now that we have generated our time polynomials, let's take a look at them.


```python
time_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>t_1</th>
      <th>t_2</th>
      <th>t_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.000000</td>
      <td>1.000000</td>
      <td>-1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.997998</td>
      <td>0.992000</td>
      <td>-0.982030</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.995996</td>
      <td>0.984016</td>
      <td>-0.964156</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.993994</td>
      <td>0.976048</td>
      <td>-0.946378</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.991992</td>
      <td>0.968096</td>
      <td>-0.928695</td>
    </tr>
  </tbody>
</table>
</div>



Notice that we have kept the scaling intact on the other time variables. From here, we can use the newly formed polynomials to model mean heterogeneity derived from trends. 

## Concluding Remarks

The purpose of this module is to summarize the problem of near-collinearity, within the context of using time polynomials to detect mean heterogeneity, and to illustrate how to use Python to employ the proposed solution given in Michaelides and Spanos (2020).  Ultimately, it is common to use trends to capture mean heterogeneity in modeling. The problem with this method, however, is that including higher order trends can invite unnecessary sensitivity of model estimates and standard errors due to the effects of near-collinearity. To avoid this issue, Michaelides and Spanos (2020) advise the re-scaling of time variables to values between [-1,1] and applying orthogonal trend polynomials when in the presence of a large number of trends.

## References

[1] Michaelides, M and Spanos, A. 2020. "On modeling heterogeneity in linear models using trend polynomials." _Economic Modeling_, 85(C), pg. 74-86.  
[2] Spanos, A. 2019. Probability Theory and Statistical Inference: Empirical Modeling With Observational Data. _Cambridge University Press._ Second Edition.
