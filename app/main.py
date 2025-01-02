import datetime
import logging
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud

from utils.prepare import *
from utils.reqs import *

# Setting up the logger
handler = RotatingFileHandler('logs/streamlit_app.log', maxBytes=2000, backupCount=10)
logging.basicConfig(level=logging.INFO, handlers=[handler])
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger = logging.getLogger("backend")
logger.addHandler(handler)

st.title("Machine Learning Dashboard")

# Initialize session state
if 'hyperparams' not in st.session_state:
    st.session_state.hyperparams = {}
if 'select_model' not in st.session_state:
    st.session_state.select_model = ''
if 'mod_list_df' not in st.session_state:
    st.session_state.mod_list_df = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = CountVectorizer()
if 'scaler' not in st.session_state:
    st.session_state.scaler = StandardScaler()
if 'experiments' not in st.session_state:
    st.session_state.experiments = []


@dataclass
class Experiment:
    date: datetime.datetime
    model_name: str
    hyper: dict[str, Any]
    auc_roc: float
    auc_roc_val: float
    fpr_val: np.ndarray
    tpr_val: np.ndarray
    fpr_train: np.ndarray
    tpr_train: np.ndarray


def logistic_hyper():
    st.title("Logistic Regression Hyperparameter Tuning")
    C = st.slider('Inverse regularization strength', 0.01, 10.0, 1.0)
    max_iter = st.slider('Maximum iterations', 100, 5000, 700)
    solver = st.selectbox('Solver', ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'])
    penalty = st.selectbox('Penalty', ['l2', 'none'])
    if st.button('Set Hyper'):
        st.session_state.hyperparams = {'C': C, 'max_iter': max_iter, 'solver': solver, 'penalty': penalty}
        st.success('### Selected Hyperparams:')
        st.json(st.session_state.hyperparams)


def decision_tree_hyper():
    st.title("Decision Tree Hyperparameter Tuning")
    criterion = st.selectbox('Criterion', ['gini', 'entropy'])
    max_depth = st.slider('Max Depth', 1, 50, 3)
    min_samples_split = st.slider('Min Samples Split', 2, 10, 2)
    if st.button('Set Hyper'):
        st.session_state.hyperparams = {'criterion': criterion, 'max_depth': max_depth,
                                        'min_samples_split': min_samples_split}
        st.success('### Selected Hyperparams:')
        st.json(st.session_state.hyperparams)


def knn_hyper():
    st.title("KNN Hyperparameter Tuning")
    n_neighbors = st.slider('Number of Neighbors', 1, 20, 5)
    weights = st.selectbox('Weights', ['uniform', 'distance'])
    algorithm = st.selectbox('Algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
    if st.button('Set Hyper'):
        st.session_state.hyperparams = {'n_neighbors': n_neighbors, 'weights': weights, 'algorithm': algorithm}
        st.success('### Selected Hyperparams:')
        st.json(st.session_state.hyperparams)


@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            logger.info(f"Data loaded successfully with shape {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            st.error("Error loading the file. Please ensure it's a valid CSV.")
            return None


def kde_plot(series):
    x = np.linspace(min(series), max(series), 1000)
    kde = gaussian_kde(series)
    y = kde(x)
    return x, y


def plot_combined_roc_curve(fpr_train,
                            tpr_train,
                            auc_score_train,
                            fpr_val,
                            tpr_val,
                            auc_score_val):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr_train, y=tpr_train, mode='lines', name=f'Train AUC = {auc_score_train:.2f}',
                             line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=fpr_val, y=tpr_val, mode='lines', name=f'Validation AUC = {auc_score_val:.2f}',
                             line=dict(color='red')))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash', color='black')))
    fig.update_layout(title='AUC-ROC Curve',
                      xaxis_title='False Positive Rate',
                      yaxis_title='True Positive Rate',
                      xaxis=dict(range=[0, 1]),
                      yaxis=dict(range=[0, 1]),
                      autosize=False, width=600, height=600)

    return fig


st.sidebar.title("Upload your dataset")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv"])

if uploaded_file:
    data = load_data(uploaded_file)
    if data is not None:
        st.write("### Dataset Preview", data.head())
        st.write("### Dataset dimension", data.shape)
        st.write("### Dataset Statistics")
        st.write(data.describe())
        try:
            # Create pie chart
            st.write('### Target Pie Chart')
            feeling_counts = data['feeling'].value_counts().reset_index()
            feeling_counts.columns = ['feeling', 'count']
            fig = px.pie(feeling_counts, names='feeling', values='count', title='Distribution of Feelings')
            st.plotly_chart(fig)
        except KeyError:
            logger.error("Target Column 'feeling' wasn't found!")
            st.error("Target Column 'feeling' wasn't found!")

        st.write("### KDE and Boxplots")
        for column in data.select_dtypes(include=['float64', 'int64']).columns:
            st.subheader(f"Variable: {column}")
            fig = make_subplots(rows=1, cols=2, subplot_titles=('KDE', 'Boxplot'))
            x, y = kde_plot(data[column])
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='KDE'), row=1, col=1)
            fig.add_trace(go.Box(y=data[column], name='Boxplot', boxmean='sd'), row=1, col=2)

            fig.update_layout(width=800, height=400, showlegend=False)
            st.plotly_chart(fig, key=f'box_{column}')

        st.write("### Correlation")
        corr_matrix = data.select_dtypes(include=['float64', 'int64']).corr()
        corr_fig = px.imshow(corr_matrix,
                             labels=dict(x="Features", y="Features", color="Correlation"),
                             x=corr_matrix.columns,
                             y=corr_matrix.columns,
                             color_continuous_scale='Viridis')

        st.plotly_chart(corr_fig)
        values = data['hour'].value_counts().sort_index().reindex(range(24), fill_value=0).values
        values = np.concatenate((values, [values[0]]))
        theta = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        theta = np.concatenate((theta, [theta[0]]))
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=np.degrees(theta),
            mode='lines+markers+text',
            text=values[:-1],
            textposition='top center',
            fill='toself',
            fillcolor='rgba(128, 0, 128, 0.25)',
            line=dict(color='purple', width=2)
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values) + 5]
                )
            ),
            showlegend=False,
            title='Freq tweets on hours'
        )
        st.plotly_chart(fig, key='t')

        positive_tweets = ' '.join(data[data['feeling'] == 1]['lemmatized'].apply(lambda x: ' '.join(x)))
        wordcloud_pos = WordCloud(width=800, height=400,
                                  colormap='YlGn', background_color='black').generate(positive_tweets)

        negative_tweets = ' '.join(data[data['feeling'] == 0]['lemmatized'].apply(lambda x: ' '.join(x)))
        wordcloud_neg = WordCloud(width=1600, height=800,
                                  colormap='Oranges_r', background_color='black').generate(negative_tweets)
        st.subheader("Word Cloud for Positive Tweets")
        fig_pos, ax_pos = plt.subplots(figsize=(12, 6))
        ax_pos.imshow(wordcloud_pos, interpolation='bilinear')
        ax_pos.axis('off')
        st.pyplot(fig_pos)

        st.subheader("Word Cloud for Negative Tweets")
        fig_neg, ax_neg = plt.subplots(figsize=(12, 6))
        ax_neg.imshow(wordcloud_neg, interpolation='bilinear')
        ax_neg.axis('off')
        st.pyplot(fig_neg)

st.sidebar.title("Models")
avail_models = st.sidebar.button("List models")
if avail_models:
    try:
        mod_list = list_request()
        if mod_list:  # Ensure the list has elements
            st.write("### Available Models")
            st.session_state.mod_list_df = pd.DataFrame(mod_list)
            st.dataframe(st.session_state.mod_list_df)
        else:
            st.warning("No models available for selection.")
    except Exception as e:
        logger.error(str(e))
        st.error(f"Error: {e}")

if st.session_state.mod_list_df is not None:
    model_index = st.sidebar.selectbox('Select model: ',
                                       st.session_state.mod_list_df['id'],
                                       index=None,
                                       format_func=lambda x:
                                       st.session_state.mod_list_df[st.session_state.mod_list_df['id'] == x][
                                           'model_type'].iloc[0],
                                       )
    if st.sidebar.button("Select"):
        res = set_model(model_index)
        if res:
            st.info("Model was selected successfully!")
            st.session_state.select_model = \
            st.session_state.mod_list_df[st.session_state.mod_list_df['id'] == model_index]['model_type'].iloc[0]
        else:
            st.error("Model wasn't selected successfully!")

if st.session_state.select_model == 'LR':
    logistic_hyper()
elif st.session_state.select_model == 'KNN':
    knn_hyper()
elif st.session_state.select_model == 'DT':
    decision_tree_hyper()

st.sidebar.title("Run Experiment")
val_fraction = st.sidebar.slider('Valid fraction', 0.1, 0.9, 0.1)
if st.sidebar.button("Run"):
    try:
        data = load_data(uploaded_file)
        X_train, y_train, X_val, y_val = prepare_dataset(data, val_fraction)
        X_train, X_val = prepare_dataset_2(X_train, X_val, st.session_state.vectorizer,
                                           st.session_state.scaler)
        with st.spinner():
            auc_roc, tpr_train, fpr_train, auc_roc_val, tpr_val, fpr_val = train_request(X_train,
                                                                                         y_train,
                                                                                         X_val,
                                                                                         y_val,
                                                                                         st.session_state.hyperparams)
        st.success("Model was trained successfully!")
        experiment = Experiment(
            date=datetime.datetime.now(),
            model_name=st.session_state.select_model,
            hyper=st.session_state.hyperparams,
            auc_roc=auc_roc,
            auc_roc_val=auc_roc_val,
            fpr_val=fpr_val,
            tpr_val=tpr_val,
            fpr_train=fpr_train,
            tpr_train=tpr_train
        )
        st.session_state.experiments.append(experiment)
        st.plotly_chart(plot_combined_roc_curve(fpr_train, tpr_train, auc_roc,
                                                fpr_val, tpr_val, auc_roc_val))

    except Exception as e:
        logger.error(f'Error: {e}')
        st.error(f'Error: {e}')

st.sidebar.title("Upload inference data")
inf_file = st.sidebar.file_uploader("Choose an inference file", type=["csv"])

if inf_file:
    try:
        data = pd.read_csv(uploaded_file)
        logger.info(f"Data loaded successfully with shape {data.shape}")
        X_test = prepare_dataset_3(data, st.session_state.vectorizer,
                                   st.session_state.scaler)
        y_pred, y_prob = inference_request(X_test)
        result = pd.DataFrame(data={'y': y_pred, 'P': y_prob})
        st.dataframe(result)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        st.error("Error loading the file. Please ensure it's a valid CSV.")

st.sidebar.title("Compare Experiments")
selected_experiments = []
for idx, experiment in enumerate(st.session_state.experiments):
    exp_label = f"{experiment.date.strftime('%Y-%m-%d %H:%M:%S')} [{experiment.model_name}]"
    if st.sidebar.checkbox(exp_label, key=f"exp_{idx}"):
        selected_experiments.append(experiment)

if selected_experiments:
    comparison_fig = make_subplots(rows=1, cols=2,
                                   subplot_titles=("ROC Curve (train)", "ROC Curve (valid)"),
                                   horizontal_spacing=0.1)
    color_map = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']
    for i, exp in enumerate(selected_experiments):
        comparison_fig.add_trace(go.Scatter(x=exp.fpr_train,
                                            y=exp.tpr_train,
                                            mode='lines',
                                            name=f'Train AUC ({exp.model_name}) = {exp.auc_roc:.2f}',
                                            line=dict(color=color_map[i % len(color_map)])), row=1, col=1)
        comparison_fig.add_trace(go.Scatter(x=exp.fpr_val, y=exp.tpr_val,
                                            mode='lines',
                                            name=f'Validation AUC ({exp.model_name}) = {exp.auc_roc_val:.2f}',
                                            line=dict(color=color_map[i % len(color_map)])), row=1, col=2)

    comparison_fig.update_xaxes(title_text="False Positive Rate", range=[0, 1], row=1, col=1)
    comparison_fig.update_yaxes(title_text="True Positive Rate", range=[0, 1], row=1, col=1)

    comparison_fig.update_xaxes(title_text="False Positive Rate", range=[0, 1], row=1, col=2)
    comparison_fig.update_yaxes(title_text="True Positive Rate", range=[0, 1], row=1, col=2)

    comparison_fig.update_layout(title_text='ROC Curves', showlegend=True)
    st.plotly_chart(comparison_fig)
