import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
from sklearn import linear_model
import altair as alt


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
    df = df[(df.z < 6) & (df.z > 2) & (df.y < 10) & (df.table > 50) &
            (df.table < 70) & (df.depth > 55) & (df.depth < 70)]
    df.drop('Unnamed: 0', axis=1, inplace=True)

    cat = df.columns[df.dtypes == 'object']
    num = df.columns[((df.dtypes == 'int64') | (df.dtypes == 'float64'))]

    choise = st.radio("Выберите тип признака:", ['Категориальный', 'Количественный'])
    if choise == 'Категориальный':
        st.dataframe(df[cat].head(), height=200)
    else:
        st.dataframe(df[num].head(), height=200)

    """
    Для того чтобы посмотреть на распределение каждого из числовых признаков, выберите его ниже:
    """

    choise = st.radio("Выберите признак", num)
    c = ['b', 'r', 'g']

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
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
    feat = ['cut', 'color', 'clarity']
    codes = [cut_code, color_code, clarity_code]

    fig, ax = plt.subplots(3, 1, figsize=(15, 10))

    for i in range(3):
        ax[i].pie([np.sum(np.array(df[feat[i]]) == j) for j in codes[i]], labels=list(codes[i]),
              autopct="%0.0f%%", textprops={"fontsize": 6}, pctdistance=0.87)

        ax[i].set_title(f"Круговая диаграмма для признака {feat[i]}")

    st.pyplot(fig)

    """
    Всегда интересно взглянуть на наличие зависимостей в данных.
    А самые интересные зависимости - это конечно связь цены алмаза и его характеристик.
    Поэтому вы можете выбрать одну из зависимых количественных переменных 
    и посмотреть на диаграмму рассеяния для нее.
    """

    choise_x = st.radio("Выберите признак, для которого будет строиться диаграмма рассеяния:", num.delete(3))

    """
    Дополнительно возможно выбрать одну из категориальных переменных чтобы посмотреть на их связь
    с ценой в совокупности с количественной. Если вы не хотите этого, то оставьте no.
    """

    choise_hue = st.radio("Выберите признак, который будет индкаторным на диаграмме:", np.append('no', cat))

    """
    Прекрасно, все почти готово. Если вы выбрали категориальную переменную,
    то диаграмма будет анимированной: каждые n секунд она будет сменяться другой диаграммой
    для следующего значения категориальной переменной. Вы можете выбрать удобный вам временной промежуток.
    Заметьте, что для построения диаграммы требуется некоторое время.
    """

    number = st.slider("Выберите время:", 1, 30, 8)

    """
    Осталось нажать кнопку и все заработает. Когда вам наскучит, просто поменяйте один из параметров
    и анимация пропадет. Без этого увидеть контент дальше не получится.
    """

    start_stop = st.button('ТЫК')

    if start_stop:

        model = linear_model.LinearRegression()
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
    Из графика выше видно, что цена нелинейно зависит от количества карат,
    и что на цену оказывают сильное влияние color и clarity, а cut почти у всех алмазов хороший.
    Поэтому особенно интересно взглянуть на распределние color и clarity в группе алмазов с разной ценой.
    """

    feat = ['color', 'clarity']
    lim = [6300, 7500]

    for i in range(2):

        fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        fig.tight_layout()
        fig.suptitle(f"Цена алмаза для {feat[i]}", y=1.1, fontsize=14)
        l_h = [df[df.price < df.price.median()], df[df.price >= df.price.median()]]
        words = ['Ниже', 'Выше']

        for j in range(2):

            df_low = df[df.price < df.price.median()]
            ax[j].set_title(f"{words[j]} среднего")
            sns.barplot(y=np.unique(l_h[j][feat[i]], return_counts=True)[0],
                        x=np.unique(l_h[j][feat[i]], return_counts=True)[1], ax=ax[j], orient='h')
            ax[j].invert_xaxis()

        ax[1].set_xlim(0, lim[i])
        ax[0].set_xlim(lim[i])

        st.pyplot(fig)                              

    """
    Здесь уже совершенно четко видно, что цвета H, I, J преобладают среди алмазов большей цены,
    а D и Е - меньшей.
    """

    """
    Заметно, что VVS1, VVS2, VS2, VS1 преобладают в нижней ценовой категории, а SI2 и SI1 - в высшей.
    """

    """
    Но это все имеет скорее экономические приложения, а вдруг мы геологи и нам интересно,
    а как вообще размер carat связан с цветом алмаза.
    Гистограмма интерактивная: если потыкать на легенду,
    то можно увидеть гистограммы carat для определенного цвета:
    """

    max_len_2 = len(df[df.color == "G"])
    matrix = []
    colors = ["G", "F", "E", "H", "D", "I", "J"]

    for val in colors:

        carat = np.array(df[df.color == val].carat)
        matrix.append(np.hstack((carat, np.array([None] * (max_len_2 - len(carat))))))

    matrix = np.array(matrix).T
    source = pd.DataFrame(matrix, columns=colors)
    selection = alt.selection_multi(fields=['Color'], bind='legend')

    ### FROM: (https://altair-viz.github.io/gallery/layered_histogram.html)

    chart = alt.Chart(source, width=740, height=500).transform_fold(
        colors,
        as_=['Color', 'Carat']
    ).mark_area(
        opacity=0.3,
        interpolate='step'
    ).encode(
        alt.X('Carat:Q', bin=alt.Bin(maxbins=20)),
        alt.Y('count()', stack=None),
        alt.Color('Color:N'),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.01))
    ).add_selection(
        selection
    )

    ### END FROM

    st.altair_chart(chart)

    """
    Вообще, если внимательно посмотреть на гистограммы, но получится заметить,
    что алмазы с большим весом - это алмазы с цветами H, I, J.
    А вот маленькие алмазы - в основном D, E, F.
    """
    """
    Поэтому можно рассмотреть отдельные групповые гистограммы для каждого из значений признака color.
    За значение, отвечающее за разделение алмазов на "большие" и "маленькие" можно взять, к примеру,
    размер в 2 карата.
    """
    n = 2
    source = df.copy()
    d = {1: 'bigger', 0: 'smaller'}
    source['bigger'] = list(map(lambda x: d[x], list(df.carat > n)))

    ### FROM: (https://altair-viz.github.io/gallery/grouped_bar_chart.html)

    chart = alt.Chart(source, width=70, height=500).mark_bar().encode(
        x='bigger:O',
        y='count(bigger):Q',
        color='bigger:N',
        column='color:N'
    )

    ### END FROM

    st.altair_chart(chart)

    """
    Здесь уже явно видно, что среди некоторых цветов больше больших алмазов, чем среди других.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
    """
