

def distribution(df, feature, xlbl, style):
    # Fonction qui affiche le feature d'un df selon le style specifié
    # pré-requis : pandas, seaborn, numpy, matplotlib
    # input : df = DataFrame
    #        feature = feature
    #        xlbl = label axe x
    # args:   style = box, violin, dist
    # output : graphe

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

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
    

