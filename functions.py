import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_score as sisc
from sklearn.metrics import davies_bouldin_score as dbsc

def distribution(df, feature, xlbl, style):
    """
    Fonction qui affiche le feature d'un df selon le style specifié
    pré-requis : pandas, seaborn, numpy, matplotlib
    input : df = DataFrame
           feature = feature
           xlbl = label axe x
    args:   style = box, violin, dist
    output : graphe
    """

    plt.figure(figsize=(11, 4))
    x = df.loc[~df[feature].isnull(),
               feature]
    x = np.array(x)
    if style == 'box':
        sns.boxplot(data=x, orient='h')
    elif style == 'violin':
        sns.violinplot(data=x, orient='h')
    elif style == 'dist':
        sns.distplot(x)
    plt.title('Distribution of {}'.format(feature))
    plt.xlabel(xlbl)
    plt.yticks([])
    plt.show()
    
    
def charm_price(df, price_feature):
    """
    fonction qui isole la terminaison "9" des prix affiché, dite "charm price"
    input : df = DataFrame
            price_feature = feature
    output : le df avec une nouvelle colonne "charm_price"
    isoler les terminaisons (Centavos, Real)
    """    
    df['last_centavos'] = df[price_feature] % 1
    df['last_real'] = df[price_feature] % 10 - df[price_feature] % 1
    df['last_ten_real'] = ((df[df[price_feature] % 10 == 0][price_feature]) / 10) % 10 
    # convertir en info booléen "is_charm" pour unités et décimales
    df['charm_centavos'] =\
        df['last_centavos'].map(
        lambda x: 0 if x < 0.88 else 1)
    df['charm_real'] =\
        df['last_real'].map(
        lambda x: 0 if x != 9 else 1)
    df['charm_ten_real'] =\
        df['last_ten_real'].map(
        lambda x: 0 if x != 9 else 1)
    # déterminer un booléen "charm_price" si l'une des terminaisons est "charm"
    df['charm_price'] = df['charm_real'] + df['charm_centavos'] + df['charm_ten_real']
    df['charm_price'] =\
        df['charm_price'].map(
        lambda x: 1 if x >= 1 else 0)
    # retirer les colonnes obsolètes
    Drop = ['last_centavos',
            'last_real',
            'last_ten_real',
            'charm_centavos',
            'charm_real',
            'charm_ten_real']
    df.drop(Drop, axis=1, inplace=True)
    return(df)


def haversine(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371):
    """
    seen on stackoverflow.com

    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees or in radians)

    All (lat, lon) coordinates must have numeric dtypes and be of equal length.

    """
    if to_radians:
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

    a = np.sin((lat2-lat1)/2.0)**2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2.0)**2

    return earth_radius * 2 * np.arcsin(np.sqrt(a))

# the perfect heatmap

def heatmap(x, y, size, color):
    """
    built out of a tutorial seen on towardsdatascience.com

    Enhance heatmap for feature correlation observation

    """
    
    fig, ax = plt.subplots(figsize=(20, 20))
    # Mapping from column names to integer coordinates
    x_labels = [v for v in x.unique()]
    y_labels = [v for v in y.unique()]
    x_to_num = {p[1]: p[0] for p in enumerate(x_labels)}
    y_to_num = {p[1]: p[0] for p in enumerate(y_labels)}
    size_scale = 500
    # Use 256 colors for the diverging color palette
    n_colors = 256
    # Create the palette
    palette = sns.diverging_palette(20, 220, n=n_colors)
    # Range of values mapped to the palt, i.e. min and max poss corr
    color_min, color_max = [-1, 1]

    def value_to_color(val):
        # pos of value in input range, relative to length of input range
        val_position = float((val - color_min)) / (color_max - color_min)
        # target index in the color palette
        ind = int(val_position * (n_colors - 1))
        return palette[ind]

    # Setup a 1x40 Grid
    plot_grid = plt.GridSpec(1, 40, hspace=0.2, wspace=0.1)
    # Use the leftmost 39 columns of the grid for the main plot
    ax = plt.subplot(plot_grid[:, :-1])

    ax.scatter(
        x=x.map(x_to_num),  # Use mapping for x
        y=y.map(y_to_num),  # Use mapping for y
        s=size * size_scale,  # Vector sq sizes
        c=color.apply(value_to_color),  # Vector sq color values
        marker='s'  # Use square as scatterplot marker
    )

    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)
    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])

    # Add color legend on the right side of the plot
    ax = plt.subplot(plot_grid[:, -1])  # Use the rightmost column of the plot

    # Fixed x coordinate for the bars
    col_x = [0]*len(palette)
    # y coordinates for each of the n_colors bars
    bar_y = np.linspace(color_min, color_max, n_colors)

    bar_height = bar_y[1] - bar_y[0]
    ax.barh(
        y=bar_y,
        width=[5]*len(palette),  # Make bars 5 units wide
        left=col_x,  # Make bars start at 0
        height=bar_height,
        color=palette,
        linewidth=0
    )
    # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
    ax.set_xlim(1, 2)
    # Hide grid
    ax.grid(False)
    # Make background white
    ax.set_facecolor('white')
    # Remove horizontal ticks
    ax.set_xticks([])
    # Show vertical ticks for min, middle and max
    ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3))
    # Show vertical ticks on the right
    ax.yaxis.tick_right()


def heatmap2(x, y, size, color, a, b):
    """
    built out of a tutorial seen on towardsdatascience.com

    Enhance heatmap for feature correlation observation

    """
    
    fig, ax = plt.subplots(figsize=(a, b))
    # Mapping from column names to integer coordinates
    x_labels = [v for v in x.unique()]
    y_labels = [v for v in y.unique()]
    x_to_num = {p[1]: p[0] for p in enumerate(x_labels)}
    y_to_num = {p[1]: p[0] for p in enumerate(y_labels)}
    size_scale = 500
    # Use 256 colors for the diverging color palette
    n_colors = 256
    # Create the palette
    palette = sns.diverging_palette(20, 220, n=n_colors)
    # Range of values mapped to the palt, i.e. min and max poss corr
    color_min, color_max = [-1, 1]

    def value_to_color(val):
        # pos of value in input range, relative to length of input range
        val_position = float((val - color_min)) / (color_max - color_min)
        # target index in the color palette
        ind = int(val_position * (n_colors - 1))
        return palette[ind]

    # Setup a 1x40 Grid
    plot_grid = plt.GridSpec(1, 10, hspace=0.2, wspace=0.1)
    # Use the leftmost 9 columns of the grid for the main plot
    ax = plt.subplot(plot_grid[:, :-1])

    ax.scatter(
        x=x.map(x_to_num),  # Use mapping for x
        y=y.map(y_to_num),  # Use mapping for y
        s=size * size_scale,  # Vector sq sizes
        c=color.apply(value_to_color),  # Vector sq color values
        marker='s'  # Use square as scatterplot marker
    )

    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels)
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)
    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])

    # Add color legend on the right side of the plot
    ax = plt.subplot(plot_grid[:, -1])  # Use the rightmost column of the plot

    # Fixed x coordinate for the bars
    col_x = [0]*len(palette)
    # y coordinates for each of the n_colors bars
    bar_y = np.linspace(color_min, color_max, n_colors)

    bar_height = bar_y[1] - bar_y[0]
    ax.barh(
        y=bar_y,
        width=[5]*len(palette),  # Make bars 5 units wide
        left=col_x,  # Make bars start at 0
        height=bar_height,
        color=palette,
        linewidth=0
    )
    # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
    ax.set_xlim(1, 2)
    # Hide grid
    ax.grid(False)
    # Make background white
    ax.set_facecolor('white')
    # Remove horizontal ticks
    ax.set_xticks([])
    # Show vertical ticks for min, middle and max
    ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3))
    # Show vertical ticks on the right
    ax.yaxis.tick_right()


def rankgauss(data, feature):
    # crée une instance de quatiletransformer
    transformer = QuantileTransformer(n_quantiles=100,
                                      random_state=1111,
                                      output_distribution='normal')
    # adpate au feature
    transformer.fit(data[[feature]])
    # préfixe le nouveau feature
    newfeat = str('RG_' + feature)
    # ajoute le nouveau feature transformé
    data[newfeat] = transformer.transform(data[[feature]])
    # retire le feature dans sa version d'origine
    # data.drop(feature, axis=1, inplace=True)
    # retourne le nouveau dataframe
    return data

def scaled_rankgauss(data, feature):
    # crée une instance de quatiletransformer
    transformer = QuantileTransformer(n_quantiles=100,
                                      random_state=1111,
                                      output_distribution='normal')
    scaler = StandardScaler()
    # adpate au feature - quantile
    transformer.fit(data[[feature]])
    # préfixe le nouveau feature
    newfeat = str('RG_' + feature)
    # ajoute le nouveau feature transformé
    data[newfeat] = transformer.transform(data[[feature]])
    # adapte au feature - scale
    scaler.fit(data[[newfeat]])
    # préfixe le nouveau feature
    new_feat_scaled = str('SC_' + newfeat)
    # ajoute le nouveau feature transformé
    data[new_feat_scaled] = scaler.transform(data[[newfeat]])
    # retire le feature dans sa version d'origine
    # data.drop(feature, axis=1, inplace=True)
    # retourne le nouveau dataframe
    return data


def dbsc_forward_selection(n, data, score_threshold):
    """
    give the "best features",
    according to davies-bouldin index
    under a given score_threshold
    for n n_clusters
    pre-requisite : 
    sklearn.cluster kmeans
    sklearn.metrics davies_bouldin_score as dbsc
    """

    initial_features = data.columns.tolist()
    best_features = []
    stored_scores = []
    kmeans = KMeans(n_clusters=n)
    while (len(initial_features) > 0):
        remaining_features = list(set(initial_features)-set(best_features))
        new_dbsc = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            new_dbsc[new_column] = dbsc(
                data[best_features+[new_column]],
                kmeans.fit_predict(data[best_features+[new_column]])
            )
        min_dbsc = new_dbsc.min()
        stored_scores.append(min_dbsc)
        if(min_dbsc < score_threshold):
            best_features.append(new_dbsc.idxmin())
        else:
            break
    print('nclusters : ' + str(n))
    print(best_features)
    print(stored_scores[:-1])
    return


def sisc_forward_selection(n, data, score_threshold):
    """
    give the "best features",
    according to silhouette_score
    over a given score_threshold
    for n n_clusters
    pre-requisite : 
    sklearn.cluster kmeans
    sklearn.metrics silhouette_score as sisc
    """

    initial_features = data.columns.tolist()
    best_features = []
    stored_scores = []
    kmeans = KMeans(n_clusters=n)
    while (len(initial_features) > 0):
        remaining_features = list(set(initial_features)-set(best_features))
        new_sisc = pd.Series(index=remaining_features)
        for new_column in remaining_features:
            new_sisc[new_column] = sisc(
                data[best_features+[new_column]],
                kmeans.fit_predict(data[best_features+[new_column]])
            )
        max_sisc = new_sisc.max()
        stored_scores.append(max_sisc)
        if(max_sisc > score_threshold):
            best_features.append(new_sisc.idxmax())
        else:
            break
    print('nclusters : ' + str(n))
    print(best_features)
    print(stored_scores[:-1])
    return