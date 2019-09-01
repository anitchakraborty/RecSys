import pandas as pd 
import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from sklearn.preprocessing import OneHotEncoder


def rec(ratingspath, moviespath):
    df=pd.read_csv(ratingspath)
    movie_titles = pd.read_csv(moviespath)
    movie_titles = movie_titles.head(1040)
    df = pd.merge(df, movie_titles, on='movieId')
    ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
    ratings['number_of_ratings'] = df.groupby('title')['rating'].count()
    movie_matrix = df.pivot_table(index='userId', columns='title', values='rating',fill_value=0)
    ALA_user_rating = movie_matrix['Aladdin (1992)']
    ADGH2_user_rating = movie_matrix['All Dogs Go to Heaven 2 (1996)']
    similar_to_ALA=movie_matrix.corrwith(ALA_user_rating)
    similar_to_ADGH2=movie_matrix.corrwith(ADGH2_user_rating)
    corr_ALA = pd.DataFrame(similar_to_ALA, columns=['Correlation'])
    corr_ALA = corr_ALA.join(ratings['number_of_ratings']).sort_values('Correlation',ascending=False)
    return corr_ALA

if __name__ == '__main__':
    print(rec('D:\\Executable\\ml-20m\\ratings1.csv', 'D:\\Executable\\ml-20m\\movies.csv'))
    

