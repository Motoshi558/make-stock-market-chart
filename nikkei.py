#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


get_ipython().system('pip install pandas_datareader')


# In[1]:


from pandas_datareader import data
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
pd.core.common.is_list_like = pd.api.types.is_list_like


# In[2]:


start = '2019-06-01'
end = '2020-06-01'
 
df = data.DataReader('^N225', 'yahoo', start, end)


# In[3]:


df.head(10)


# In[4]:


df.index


# In[5]:


date=df.index
price=df['Adj Close']


# In[6]:


plt.plot(date,price)


# In[7]:


plt.figure(figsize=(30, 10))
plt.plot(date,price)


# In[8]:


plt.figure(figsize=(30, 10))
plt.plot(date,price,label='Nikkei225')
plt.legend()


# In[9]:


plt.figure(figsize=(30, 10))
plt.plot(date,price,label='Nikkei225')
plt.title('N225',color='blue',backgroundcolor='white',size=40,loc='center')
plt.legend()


# In[10]:


plt.figure(figsize=(30, 10))
plt.plot(date,price,label='Nikkei225')
plt.title('N225',color='blue',backgroundcolor='white',size=40,loc='center')
plt.xlabel('date',color='black',size=30)
plt.ylabel('price',color='black',size=30)
plt.legend()


# In[11]:


span01=5
span02=25
span03=50

df['sma01'] = price.rolling(window=span01).mean()
df['sma02'] = price.rolling(window=span02).mean()
df['sma03'] = price.rolling(window=span03).mean()


# In[12]:


pd.set_option('display.max_rows', None)
df.head(100)


# In[13]:


plt.figure(figsize=(30, 10))
plt.plot(date,price,label='Nikkei225')
plt.plot(date,df['sma01'],label='sma01')
plt.plot(date,df['sma02'],label='sma02')
plt.plot(date,df['sma03'],label='sma03')

plt.title('N225',color='blue',backgroundcolor='white',size=40,loc='center')
plt.xlabel('date',color='black',size=30)
plt.ylabel('price',color='black',size=30)

plt.legend()


# In[14]:


#配色の参考サイト
#https://colorhunt.co/palette/184189


# In[15]:


plt.figure(figsize=(30, 10))
plt.plot(date,price,label='Nikkei225',color='#99b898')
plt.plot(date,df['sma01'],label='sma01',color='#e84a5f')
plt.plot(date,df['sma02'],label='sma02',color='#ff847c')
plt.plot(date,df['sma03'],label='sma03',color='#feceab')

plt.title('N225',color='white',backgroundcolor='grey',size=30,loc='center')
plt.xlabel('date',color='grey',size=20)
plt.ylabel('price',color='grey',size=20)

plt.legend()


# In[16]:


plt.figure(figsize=(30, 15))
plt.bar(date,df['Volume'],label='Volume',color='grey')

plt.legend()


# In[17]:


plt.figure(figsize=(30, 15))
plt.subplot(2,1,1)

plt.plot(date,price,label='Close',color='#99b898')
plt.plot(date,df['sma01'],label='sma01',color='#e84a5f')
plt.plot(date,df['sma02'],label='sma02',color='#ff847c')
plt.plot(date,df['sma03'],label='sma03',color='#feceab')
plt.legend()

plt.subplot(2,1,2)
plt.bar(date,df['Volume'],label='Volume',color='grey')
plt.legend()


# In[18]:


#リクルートホールディングス
df = data.DataReader('6098.JP', 'stooq')


# In[19]:


df.head()


# In[20]:


df.index.min()


# In[21]:


df.index.max()


# In[22]:


df.tail()


# In[23]:


df = df.sort_index()


# In[24]:


df.head(10)


# In[25]:


df.index>='2019-06-01 00:00:00'


# In[26]:


df[df.index>='2019-06-01 00:00:00']


# In[27]:


df[df.index<='2020-05-01 00:00:00']


# In[28]:


df[(df.index>='2019-06-01 00:00:00') & (df.index<='2020-05-01 00:00:00')]


# In[29]:


df = df[(df.index>='2019-06-01 00:00:00') & (df.index<='2020-05-01 00:00:00')]


# In[30]:


#リクルートホールディングス
date=df.index
price=df['Close']

span01=5
span02=25
span03=50

df['sma01'] = price.rolling(window=span01).mean()
df['sma02'] = price.rolling(window=span02).mean()
df['sma03'] = price.rolling(window=span03).mean()

plt.figure(figsize=(30, 15))
plt.subplot(2,1,1)

plt.plot(date,price,label='Close',color='#99b898')
plt.plot(date,df['sma01'],label='sma01',color='#e84a5f')
plt.plot(date,df['sma02'],label='sma02',color='#ff847c')
plt.plot(date,df['sma03'],label='sma03',color='#feceab')
plt.legend()

plt.subplot(2,1,2)
plt.bar(date,df['Volume'],label='Volume',color='grey')
plt.legend()


# In[ ]:





# In[31]:


start = '2019-06-01'
end = '2020-05-01'
company_code = '6502.JP'


# In[32]:


#ユニクロやGUなどがグループ会社のファーストリテイリング
df = df[(df.index>=start) & (df.index<=end)]
df = data.DataReader(company_code, 'stooq')
df = df[(df.index>=start) & (df.index<=end)]

date=df.index
price=df['Close']

span01=5
span02=25
span03=50

df['sma01'] = price.rolling(window=span01).mean()
df['sma02'] = price.rolling(window=span02).mean()
df['sma03'] = price.rolling(window=span03).mean()

plt.figure(figsize=(30, 15))
plt.subplot(2,1,1)

plt.plot(date,price,label='Close',color='#99b898')
plt.plot(date,df['sma01'],label='sma01',color='#e84a5f')
plt.plot(date,df['sma02'],label='sma02',color='#ff847c')
plt.plot(date,df['sma03'],label='sma03',color='#feceab')
plt.legend()

plt.subplot(2,1,2)
plt.bar(date,df['Volume'],label='Volume',color='grey')
plt.legend()


# In[33]:


def company_stock(start,end,company_code):
    df = data.DataReader(company_code, 'stooq')
    df = df[(df.index>=start) & (df.index<=end)]

    date=df.index
    price=df['Close']

    span01=5
    span02=25
    span03=50

    df['sma01'] = price.rolling(window=span01).mean()
    df['sma02'] = price.rolling(window=span02).mean()
    df['sma03'] = price.rolling(window=span03).mean()

    plt.figure(figsize=(20, 10))
    plt.subplot(2,1,1)

    plt.plot(date,price,label='Close',color='#99b898')
    plt.plot(date,df['sma01'],label='sma01',color='#e84a5f')
    plt.plot(date,df['sma02'],label='sma02',color='#ff847c')
    plt.plot(date,df['sma03'],label='sma03',color='#feceab')
    plt.legend()

    plt.subplot(2,1,2)
    plt.bar(date,df['Volume'],label='Volume',color='grey')
    plt.legend()


# In[34]:


company_stock('2019-06-01','2020-06-01','6502.JP')


# In[35]:


company_stock('2017-01-01','2020-06-01','6502.JP')


# In[36]:


company_stock('2017-01-01','2020-06-01','7203.jp')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # ▲ ここまで

# In[ ]:


fig, (ax1, ax2) = plt.subplots(2,1,figsize=(30, 15), gridspec_kw = {'height_ratios':[6,2]})

ax1.plot(date,price,label='Close',color='#99b898')
ax1.plot(date,df['sma01'],label='sma01',color='#e84a5f')
ax1.plot(date,df['sma02'],label='sma02',color='#ff847c')
ax1.plot(date,df['sma03'],label='sma03',color='#feceab')

ax2.bar(date,df['Volume'],label='Volume',color='grey')

ax1.legend()
ax2.legend()


# In[ ]:





# In[ ]:





# In[ ]:


# !conda install -c quantopian ta-lib -y


# In[ ]:


import talib as ta


# In[ ]:


macd, macdsignal, macdhist = ta.MACD(np.array(df.Close), fastperiod=span01, slowperiod=span02, signalperiod=span01)
df['macd'] = macd
df['macdsignal'] = macdsignal
df['macdhist'] = macdhist


# In[ ]:


xmin = df.index.min()
xmax = df.index.max()


# In[ ]:


plt.figure(figsize=(30, 10))
plt.fill_between(date, macdhist, color = 'grey', alpha=0.5, label='Histogram')
plt.hlines([0],xmin,xmax,"gray",linestyles="dashed")


# In[ ]:


savename = 'N225.png'
plt.savefig(savename,dpi=300)


# In[ ]:





# # いろいろなグラフ

# In[ ]:


import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

# Sort colors by hue, saturation, value and name.
by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                for name, color in colors.items())
sorted_names = [name for hsv, name in by_hsv]

n = len(sorted_names)
ncols = 4
nrows = n // ncols + 1

fig, ax = plt.subplots(figsize=(20, 10))

# Get height and width
X, Y = fig.get_dpi() * fig.get_size_inches()
h = Y / (nrows + 1)
w = X / ncols

for i, name in enumerate(sorted_names):
    col = i % ncols
    row = i // ncols
    y = Y - (row * h) - h

    xi_line = w * (col + 0.05)
    xf_line = w * (col + 0.25)
    xi_text = w * (col + 0.3)

    ax.text(xi_text, y, name, fontsize=(h * 0.8),
            horizontalalignment='left',
            verticalalignment='center')

    ax.hlines(y + h * 0.1, xi_line, xf_line,
              color=colors[name], linewidth=(h * 0.6))

ax.set_xlim(0, X)
ax.set_ylim(0, Y)
ax.set_axis_off()



fig.subplots_adjust(left=0, right=1,
                    top=1, bottom=0,
                    hspace=0, wspace=0)

plt.show()


# In[ ]:





# In[ ]:


import matplotlib.pyplot as plt

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'A', 'B', 'C', 'D'
sizes = [15, 30, 45, 10]
explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots(figsize=(20, 10))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# Random test data
np.random.seed(123)
all_data = [np.random.normal(0, std, 100) for std in range(1, 4)]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 8))

# rectangular box plot
bplot1 = axes[0].boxplot(all_data,
                         vert=True,   # vertical box aligmnent
                         patch_artist=True)   # fill with color

# notch shape box plot
bplot2 = axes[1].boxplot(all_data,
                         notch=True,  # notch shape
                         vert=True,   # vertical box aligmnent
                         patch_artist=True)   # fill with color

# fill with colors
colors = ['pink', 'lightblue', 'lightgreen']
for bplot in (bplot1, bplot2):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

plt.figure(figsize=(30, 10))
        
# adding horizontal grid lines
for ax in axes:
    ax.yaxis.grid(True)
    ax.set_xticks([y+1 for y in range(len(all_data))], )
    ax.set_xlabel('xlabel')
    ax.set_ylabel('ylabel')



# add x-tick labels
plt.setp(axes, xticks=[y+1 for y in range(len(all_data))],
         xticklabels=['x1', 'x2', 'x3', 'x4'])

plt.show()


# In[ ]:


df.diff().hist(color='k', alpha=0.5, bins=50,figsize=(20, 10))
plt.show()


# In[ ]:


import matplotlib
print(matplotlib.__version__)
# 1.5.1

import numpy as np
from scipy.stats import multivariate_normal

#for plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

m = 2 #dimension
mean = np.zeros(m)
sigma = np.eye(m)


# In[ ]:


N = 1000
x1 = np.linspace(-5, 5, N)
x2 = np.linspace(-5, 5, N)

X1, X2 = np.meshgrid(x1, x2)
X = np.c_[np.ravel(X1), np.ravel(X2)]

Y_plot = multivariate_normal.pdf(x=X, mean=mean, cov=sigma)
Y_plot = Y_plot.reshape(X1.shape)

fig = plt.figure(figsize=(30, 10))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X1, X2, Y_plot, cmap='bwr', linewidth=0)
fig.colorbar(surf)
ax.set_title("Surface Plot")
fig.show()


# # ▼株

# In[ ]:


df['rsi_5'] = ta.RSI(np.array(df.Close), timeperiod=5)


# In[ ]:


df['rsi_20'] = ta.RSI(np.array(df.Close), timeperiod=20)


# In[ ]:





# In[ ]:


xmin = df.index.min()
xmax = df.index.max()


# In[ ]:


fig, (ax1, ax2,ax3) = plt.subplots(3,1,figsize=(30, 15), gridspec_kw = {'height_ratios':[6,2,2]})

ax1.plot(date,price,label='Close',color='#99b898')
ax1.plot(date,df['sma01'],label='sma01',color='#e84a5f')
ax1.plot(date,df['sma02'],label='sma02',color='#ff847c')
ax1.plot(date,df['sma03'],label='sma03',color='#feceab')

# ax2.plot(date,df['rsi_5'],label='rsi_5')
ax2.plot(date,df['rsi_20'],label='rsi_20')
ax2.hlines([30,50,70],xmin,xmax,"gray",linestyles="dashed")
ax2.set_ylim([0,100])

ax3.fill_between(date, macdhist, color = 'grey', alpha=0.5, label='Histogram')
ax3.hlines([0],xmin,xmax,"gray",linestyles="dashed")

ax1.legend()
ax2.legend()
ax3.legend()


# In[ ]:


fig = plt.figure(figsize=(30, 10))
fig.plot(date,s1,label='Close')


# In[ ]:


fig = plt.figure(figsize=(30, 10))
ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(date,s1,label='Close')


# In[ ]:





# In[ ]:





# In[ ]:


get_ipython().system('pip install cufflinks')


# In[ ]:


import cufflinks as cf


# In[ ]:


qf.add_sma([10,20],width=2,color=['green','lightgreen'],legendgroup=True)
qf.add_rsi(periods=20,color='java')
qf.add_bollinger_bands(periods=20,boll_std=2,colors=['magenta','grey'],fill=True)
qf.add_macd()
qf.iplot()


# In[ ]:





# In[ ]:


import plotly.offline as pyo
import cufflinks as cf
import numpy as np


# In[ ]:


pyo.init_notebook_mode()


# In[ ]:


qf = cf.QuantFig(df, name='日経 225')

pyo.iplot(
    qf.iplot(asFigure=True)
)


# In[ ]:





# In[ ]:


df.to_csv('N225.csv')


# In[ ]:




