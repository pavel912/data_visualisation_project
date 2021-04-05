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
    """
    Наша таблица с данными достаточно большая, так что вывести целиком ее не получится.
    Зато можно посмотреть на разные типы признаков: категориальные и количественные:
    """
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    cat = df.columns[df.dtypes == 'object']
    num = df.columns[(df.dtypes == 'float64') | (df.dtypes == 'int64') ]
    choise = st.radio("Выберите тип признака", ['Категориальный', 'Количественный'])
    if choise == 'Категориальный':
        st.dataframe(df[cat[:len(cat) // 2]], height=5)
        st.dataframe(df[cat[len(cat) // 2:]], height=5)
    else:
        st.dataframe(df[num], height=5)
    #st.pyplot(fig)
