import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
with st.echo(code_location='below'):
    st.title("Проект по визуализации данных")
    """
    В данном проекте мы попытаемся визуализирвоать данные,
    характеризующие алмазы и их цену.
    """
    """
    Наша таблица с данными достаточно большая, так что вывести целиком ее не получится.
    Зато можно посмотреть на разные типы признаков: категориальные и количественные:
    """
    df = pd.read_csv('diamonds.csv')
    df = df[(df.z < 6) & (df.z > 2) & (df.y < 10) & (df.table > 50) & (df.table < 70) & (df.depth > 55) & (df.depth < 70)]
    df.drop('Unnamed: 0', axis=1, inplace=True)
    cat = df.columns[df.dtypes == 'object']
    num = df.columns[((df.dtypes == 'int64') | (df.dtypes == 'float64'))]
    choise = st.radio("Выберите тип признака", ['Категориальный', 'Количественный'])
    if choise == 'Категориальный':
        st.dataframe(df[cat].head(), height=200)
    else:
        st.dataframe(df[num].head(), height=200)
    """
    Для того чтобы посмотреть на распределение каждого из признаков, выберите его ниже:
    """
    choise = st.radio("Выберите признак", df.columns)
    c = ['b', 'r', 'g']
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.hist(df[choise], bins=20, color=c[np.where(np.array(df.columns) == choise)[0][0] % 3])
    ax.grid()
    ax.set_title(f"Pacпределение признака {choise}")
    st.pyplot(fig)
    #fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    #ax = sns.boxplot(x='clarity', y='price', data=df, whis=1)
    #st.pyplot(fig)
