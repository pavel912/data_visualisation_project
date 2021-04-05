import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

with st.echo(code_location='below'):
    st.title("Проект по визуализации данных")
    """
    В данном проекте мы попытаемся визуализирвоать данные, связанные
    с оттоком клиентов из телекоммуникационной компании Telco.
    """
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    st.line_chart(df)
    #x = np.linspace(0, 10, 500)
    #fig = plt.figure()
    #plt.plot(x, np.sin(x))
    #plt.ylim(-2, 2)
    #st.pyplot(fig)
