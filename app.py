import streamlit as st
import pandas as pd
import plotly.express as px
from utils import treemap_diagram
import plotly.graph_objects as go
import requests
import json

st.balloons()
st.title("Elon's tweets sentiment analysis")
st.sidebar.title("Описание проекта")

st.sidebar.markdown("<hr>", unsafe_allow_html=True)

st.sidebar.write(
    """
    Социальные сети являются неотъемлемой частью жизни многих людей. X (бывш. - Twitter) - одна из самых популярных 
    платформ, где каждый может общаться друг с другом, обмениваясь информацией и мнениями. Илон Маск - один из самых 
    известных предпринимателей и инноваторов в мире. Его твиты могут оказывать влияние финансовые рынки, в частности, в 
    отношении акций компаний, с которыми он связан (например, Tesla и SpaceX). Изучение тональности постов Маска может 
    дать представление о том, как общество воспринимает идеи и инициативы Илона Маска, а также о том, как он реагирует 
    на общественные настроения.
    """
)


API_URL = "http://localhost:8000"


models = {
    "DecisionTree": "DecisionTree",
    "LogisticRegression": "LogisticRegression",
    "CatBoost": "CatBoost"

}

penalty_logreg = {"l1": "l1", "l2" : "l2", "elasticnet" : "elasticnet", "none" : "none"}

def main():
    df = pd.read_csv("data/elon_musk_tweets_labeled.csv").drop(['Unnamed: 0', 'id', 'user_created', 'hashtags', 'user_name', 'is_retweet'], axis=1)

    st.image("images/elon.png", caption="Cool Elon Musk")

    eda_tab, prediction_tab = st.tabs([":mag: Анализ данных", ":crystal_ball: Предсказание тональности"])

    with eda_tab:
        st.subheader("EDA: Анализируем данные")

        st.subheader("Первые 5 строк датасета")
        st.dataframe(df.head())

        st.divider()

        st.subheader("Статистические характеристики")
        df_descr = df.describe(percentiles=[0.01, 0.05, 0.10, 0.25, 0.90, 0.95, 0.99]).round(2)
        df_descr_object = df.describe(include='object')

        st.dataframe(df_descr)
        st.dataframe(df_descr_object)

        st.divider()

        st.subheader("Корреляционный анализ")
        corr_matrix = df.corr(numeric_only=True)

        fig = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='rainbow')
        fig.update_layout(height=700, width=700)
        st.plotly_chart(fig)
        st.write('выводы')

        st.divider()

        st.subheader("Графики распределения: числовые признаки")
        float_feature = st.selectbox("Выберите вещественный признак:",
                                     options=list(df.select_dtypes(include=["int", "float"]).columns))
        st.plotly_chart(px.histogram(df, x=float_feature, color="feeling_auto",
                                     title=f"Распределение признака {float_feature}"
                                     ).update_xaxes(categoryorder="total descending"), use_container_width=True)
        st.write('выводы')

        st.divider()

        st.subheader("Графики распределения: категориальные признаки")
        cat_feature = st.selectbox("Выберите категориальный признак:",
                                   options=list(df.select_dtypes(include="object").columns))
        st.plotly_chart(px.histogram(df, x=cat_feature, color="feeling_auto",
                                     title=f"Распределение признака {cat_feature}"
                                     ).update_xaxes(categoryorder="total descending"), use_container_width=True)
        st.write('выводы')

        st.divider()
        st.subheader("Символы в твитах на treemap-диаграмме")

        values, lbl, colors = treemap_diagram()

        fig = go.Figure(go.Treemap(
            labels=lbl,
            parents=['']*len(values),
            values=values,
            textinfo='label+value',
            marker=dict(colors=colors),
        ))

        fig.update_layout(title='Treemap-диаграмма')
        fig.update_layout(
            width=1000,
            height=800
        )

        st.plotly_chart(fig)

        st.divider()

        st.subheader("Графики распределения: тональность твитов по времени")
        fig = px.histogram(df, x="date", color="feeling_auto", title="Изменение тональности твитов относительно времени публикации")
        st.plotly_chart(fig)

    with prediction_tab:
        st.subheader("Предсказываем тональности твита")

        selected_model = st.selectbox("Select a model", list(models.keys()))

        model_file = models[selected_model]

        if model_file == 'CatBoost':
            depth = st.slider("Глубина:", min_value=1, max_value=5)
            learning_rate = st.slider("learning rate:", min_value=0.01, max_value=0.1)
            params = {'depth': depth, 'learning_rate': learning_rate}
        if model_file == 'LogisticRegression':
            C = st.slider("C", min_value=0.01, max_value=10.0)
            penalty = st.selectbox("Penalty:", penalty_logreg)
            params = {'C': C, 'penalty': penalty}
        if model_file == 'DecisionTree':
            depth = st.slider("Глубина:", min_value=1, max_value=5)
            min_samples_leaf = st.slider("min_samples_leaf:", min_value=1, max_value=5)
            params = {"Глубина:": depth, "min_samples_leaf": min_samples_leaf}


        if st.button("Predict"):

            response = requests.post(API_URL + f"/{model_file}")
            st.write(f"Prediction: {response.json()}")

        if st.button("Fit"):

            response = requests.post(API_URL + f"http://localhost:8000/fit/{model_file}", json=params)

            if response.status_code == 200:
                st.success(response.json())
            else:
                st.error("Ошибка при обучении модели")




if __name__ == "__main__":
    main()
