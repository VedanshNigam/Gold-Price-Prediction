import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
from sklearn.ensemble import RandomForestRegressor
import numpy as np

df=pd.read_csv("gld_price_data.csv")
st.title ("GOLD PRICE PREDICTION")
X_train=pd.read_pickle('xtrain.pkl')
Y_train=pd.read_pickle('ytrain.pkl')
X_test=pd.read_pickle('xtest.pkl')
Y_test=pd.read_pickle('ytest.pkl')
regressor=RandomForestRegressor(n_estimators=100,random_state=2)
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

with st.sidebar:
    selected = option_menu("Navigation", ["Home", 'Price predictor','Plots','Dataset'],icons=['house','graph-up','file-bar-graph','file-ruled'],menu_icon=['grid'], default_index=0)
if selected =='Home':
   st.header("This is a Gold Price Prediction Application, created using Python and it's following modules :\n1. Pandas\n2. Sklearn\n3. Matplotlib\n4. Seaborn\n5. Streamlit" )     
   st.header("The Application utilizes the data from the following Stocks and Indices :\n1. SPX - Sequenced Packet Exchange\n2. USO - United States Oil\n3. SLV - Silver\n4. EUR/USD - Euro/US DOllar Pair")
if selected == 'Dataset'  : 
   st.header("Gold Price Data (2008-2018)")
   st.dataframe(df,width=1000,height=644)
if selected == 'Plots' :  
  st.header("GOLD PRICE LINEAR PLOT")
  fig=plt.figure(figsize=(5,3))
  plt.plot(df['GLD'][::10])
  st.pyplot(fig)
  correlation=df.corr()
  st.header("CORRELATION HEATMAP")
  fig=plt.figure(figsize=(5,3))
  sns.heatmap(correlation,cbar=True,fmt='.1f',cmap='Blues',annot=True)
  st.pyplot(fig)
  st.header("GOLD PRICE HISTOGRAM")
  fig2=plt.figure()
  plt.hist(df['GLD'],color='skyblue')
  st.pyplot(fig2)
  st.header("ACTUAL VS PREDICTED")
  y_test=list(Y_test)
  fig3=plt.figure()
  plt.plot(y_test[::5],color='blue',label='Actual')
  y_pred=list(Y_pred)
  plt.plot(y_pred[::5],color='red',label='Predicted')
  plt.legend()
  st.pyplot(fig3)

if selected == 'Price predictor':
#INPUT
 st.header("PRICE PREDICTION")
 spx=st.number_input("SPX",min_value=0.0,value=1.0,step=1.0)
 uso=st.number_input("USO",min_value=0.0,value=1.0,step=1.0)
 slv=st.number_input("SLV",min_value=0.0,value=1.0,step=1.0)
 eurusd=st.number_input("EUR/USD",min_value=0.0,value=1.0,step=1.0)
 input=pd.DataFrame(columns=['SPX','USO','SLV','EUR/USD'],dtype='float64')
 input.loc[0]=[spx,uso,slv,eurusd]
 st.write(input)
 gldprice=regressor.predict(input)
 st.header("Predicted Price : ")
 st.header("{:.2f}".format(gldprice[0]))



