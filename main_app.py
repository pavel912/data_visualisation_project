import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
from sklearn.linear_model import LinearRegression


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
    df = pd.read_csv('/home/pavel/Documents/programming_staff/PycharmProjects/my_cool_project_2021/my-cool-project-2021-main/diamonds.csv')
    df = df[(df.z < 6) & (df.z > 2) & (df.y < 10) & (df.table > 50) &
            (df.table < 70) & (df.depth > 55) & (df.depth < 70)]
    df.drop('Unnamed: 0', axis=1, inplace=True)

    cat = df.columns[df.dtypes == 'object']
    num = df.columns[((df.dtypes == 'int64') | (df.dtypes == 'float64'))]

    choise = st.radio("Выберите тип признака", ['Категориальный', 'Количественный'])
    if choise == 'Категориальный':
        st.dataframe(df[cat].head(), height=200)
    else:
        st.dataframe(df[num].head(), height=200)

    """
    Для того чтобы посмотреть на распределение каждого числовых признаков, выберите его ниже:
    """

    choise = st.radio("Выберите признак", num)
    c = ['b', 'r', 'g']
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    # ax.hist(df[choise], bins=20, color=c[np.where(np.array(df.columns) == choise)[0][0] % 3])
    ax.hist(df[choise], bins=20, color='lightblue')
    max_tick = ax.get_yticks()[-1]
    plt.plot([df[choise].mode(), df[choise].mode()],
             [0, max_tick], color="r", label="Мода")
    plt.plot([df[choise].median(), df[choise].median()],
             [0, max_tick], color="green", label="Медиана")
    plt.plot([df[choise].mean(), df[choise].mean()],
             [0, max_tick], color="black", label="Среднее")
    ax.set_xlabel(choise)
    ax.set_ylabel("Number of objects")
    ax.grid()
    ax.set_title(f"Pacпределение признака {choise}")
    plt.legend()
    st.pyplot(fig)

    """
    Для категориальных признаков часто хочется знать распределение значений признаков,
    но тут важнее не количество объектов, а их процентное соотношение:
    """
    cut_code = {'Ideal': 0, 'Premium': 1, 'Good': 2, 'Very Good': 3, 'Fair': 4}
    color_code = {'E': 0, 'I': 1, 'J': 2, 'H': 3, 'F': 4, 'G': 5, 'D': 6}
    clarity_code = {'SI2': 0, 'SI1': 1, 'VS1': 2, 'VS2': 3, 'VVS2': 4, 'VVS1': 5, 'I1': 6, 'IF': 7}
    fig, ax = plt.subplots(3, 1, figsize=(15, 10))

    ax[0].pie([np.sum(np.array(df.cut) == i) for i in cut_code], labels=list(cut_code),
              autopct="%0.0f%%", textprops={"fontsize": 6}, pctdistance=0.87)
    ax[0].set_title("Круговая диаграмма для признака cut")

    ax[1].pie([np.sum(np.array(df.color) == i) for i in color_code], labels=list(color_code),
              autopct="%0.0f%%", textprops={"fontsize": 6}, pctdistance=0.87)
    ax[1].set_title("Круговая диаграмма для признака color")

    ax[2].pie([np.sum(np.array(df.clarity) == i) for i in clarity_code], labels=list(clarity_code),
              autopct="%0.0f%%", textprops={"fontsize": 6}, pctdistance=0.87)
    ax[2].set_title("Круговая диаграмма для признака clarity")

    st.pyplot(fig)

    """
    Всегда интересно взглянуть на наличие зависимостей в данных.
    А самые интересные зависимости - это конечно связь цены алмаза и его характеристик.
    Поэтому вы можете выбрать одну из зависимых количественных переменных 
    и посмотреть на диаграмму рассеяния для нее.
    """
    choise_x = st.radio("Выберите признак, для которого будет строиться диаграмма рассеяния", num.delete(3))
    """
    Дополнительно возможно выбрать одну из категориальных переменных чтобы посмотреть на их связь
    с ценой в совокупности с количественной. Если вы не хотите этого, то оставьте no.
    """
    choise_hue = st.radio("Выберите признак, который будет индкаторным на диаграмме", np.append('no', cat))

    """
    Прекрасно, все почти готово. Если вы выбрали категориальную переменную,
    то диаграмма будет анимированной: каждые n секунд она будет сменяться другой диаграммой
    для следующего значения категориальной переменной. Вы можете выбрать удобный вам временной промежуток.
    Заметьте, что для построения диаграммы требуется некоторое время.
    """
    number = st.slider("Выберите время:", 1, 15, 8)
    """
    Осталось нажать кнопку и все заработает. Когда вам наскучит, просто поменяйте один из параметров
    и анимация пропадет. Без этого увидеть контент дальше не получится.
    """
    start_stop = st.button('ТЫК')
    if start_stop:
        model = LinearRegression()
        fig = plt.figure(figsize=(10, 7))
        plot = st.pyplot(fig)
        if choise_hue == 'no':
            X = np.array([np.array(df[choise_x])]).T
            y = np.array(df.price)
            sns.scatterplot(x=choise_x, y='price', data=df, alpha=1, s=3)

            x = np.arange(np.min(df[choise_x]), np.max(df[choise_x]), 0.2)

            model.fit(X, y)
            k = model.coef_[0]
            b = model.intercept_
            plt.plot(x, k * x + b, color='r', label="Линейная зависимость")

            model.fit(X ** 2, y)
            k = model.coef_[0]
            b = model.intercept_
            plt.plot(x, k * x ** 2 + b, color='black', label="Квадратичная зависимость")

            plt.grid()
            plt.title(f"Scatter plot для всей выборки")
            plt.ylim(0, 20000)
            plt.legend(loc=2)
            plot.pyplot(plt)
            fig = plt.figure(figsize=(10, 7))
        else:
            while True:
                for val in df[choise_hue].unique():
                    df_val = df[df[choise_hue] == val]
                    X = np.array([np.array(df_val[choise_x])]).T
                    y = np.array(df_val.price)

                    sns.scatterplot(x=choise_x, y='price', data=df_val, alpha=1, s=3)

                    x = np.arange(np.min(df_val[choise_x]), np.max(df_val[choise_x]), 0.2)

                    model.fit(X, y)
                    k = model.coef_[0]
                    b = model.intercept_
                    plt.plot(x, k * x + b, color='r', label="Линейная зависимость")

                    model.fit(X ** 2, y)
                    k = model.coef_[0]
                    b = model.intercept_
                    plt.plot(x, k * x ** 2 + b, color='black', label="Квадратичная зависимость")

                    plt.grid()
                    plt.ylim(0, 20000)
                    plt.title(f"Scatter plot для значения {val}")
                    plt.legend(loc=2)
                    plot.pyplot(plt)
                    fig = plt.figure(figsize=(10, 7))
                    time.sleep(number)

    """
    Что нибудь еще
    """
# TODO: Круговые диаграммы, boxplots, some animations maybe, моды, медианы и т.д.
