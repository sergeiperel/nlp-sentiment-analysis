import pandas as pd


df = pd.read_csv("data/elon_musk_tweets_labeled.csv").drop(['Unnamed: 0', 'id', 'user_created', 'hashtags', 'user_name', 'is_retweet'], axis=1)

df['._count'] = df['text'].str.count(r'\.')
df['!_count'] = df['text'].str.count(r'\!')
df['@_count'] = df['text'].str.count(r'\@')
df["'_count"] = df['text'].str.count(r'\'')
df[',_count'] = df['text'].str.count(r'\,')
df['/_count'] = df['text'].str.count(r'\/')
df['?_count'] = df['text'].str.count(r'\?')
df[';_count'] = df['text'].str.count(r'\;')
df['-_count'] = df['text'].str.count(r'\-')
df[')_count'] = df['text'].str.count(r'\)')
df['#_count'] = df['text'].str.count(r'\#')
df['(_count'] = df['text'].str.count(r'\(')


def treemap_diagram():

    colors = ['lightcoral', 'khaki', 'sandybrown', 'navajowhite', 'plum', 'palegreen', 'mediumaquamarine', 'lightblue', 'mediumpurple', 'orchid', 'pink', 'crimson']
    #
    labels = ['.', '!', '@', "'", ',', '/', '?', ';', '-', ')', '#', '(']

    summary_all = df.agg({
        '._count': 'sum',
        '!_count': 'sum',
        '@_count': 'sum',
        "'_count": 'sum',
        ",_count": 'sum',
        '/_count': 'sum',
        '?_count': 'sum',
        ';_count': 'sum',
        '-_count': 'sum',
        ')_count': 'sum',
        '#_count': 'sum',
        '(_count': 'sum',
    })

    perc = [f'{i / sum(summary_all.values.tolist()) * 100:5.2f}%' for i in summary_all.values.tolist()]
    lbl = [f'"{el[0]}" \n {el[1]}' for el in zip(labels, perc)]

    return summary_all.values.tolist(), lbl, colors
