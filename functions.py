import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def distribution(df, feature, xlbl, style):
    # Fonction qui affiche le feature d'un df selon le style specifié
    # pré-requis : pandas, seaborn, numpy, matplotlib
    # input : df = DataFrame
    #        feature = feature
    #        xlbl = label axe x
    # args:   style = box, violin, dist
    # output : graphe


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
    # fonction qui isole la terminaison "9" des prix affiché, dite "charm price"
    # input : df = DataFrame
    #         price_feature = feature
    # output : le df avec une nouvelle colonne "charm_price"
    # isoler les terminaisons (Centavos, Real)

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