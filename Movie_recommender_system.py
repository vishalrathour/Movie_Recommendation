#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies


# In[4]:


credits.head(1)


# In[5]:


credits.head(1)['cast'].values


# In[6]:


movies = movies.merge(credits,on='title')


# In[7]:


movies.head(1)


# In[8]:


movies.shape


# In[9]:


movies.info()


# In[10]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[11]:


movies.head(1)


# In[12]:


movies.isnull().sum()


# In[13]:


movies.shape


# In[14]:


movies.dropna(inplace=True)


# In[15]:


movies.shape


# In[16]:


movies.isnull().sum()


# In[17]:


movies.duplicated().sum()


# In[18]:


movies.iloc[0].genres


# In[19]:


import ast

ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[20]:


def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[21]:


movies['genres'] = movies['genres'].apply(convert)


# In[22]:


movies.iloc[0].genres


# In[23]:


movies.head()


# In[24]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[25]:


movies['keywords']


# In[26]:


movies.head(1)


# In[27]:


movies['cast']


# In[28]:


def convert3(obj):
    L=[]
    counter = 0
    for i in ast.literal_eval(obj):
        if counter!=3 :
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


# In[29]:


movies['cast'] = movies['cast'].apply(convert3)


# In[30]:


movies.head(1)


# In[31]:


def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


# In[32]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[33]:


movies['crew']


# In[34]:


movies.head(1)


# In[35]:


movies['overview'][0]


# In[36]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[37]:


movies.head()


# In[38]:


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[39]:


movies.head()


# In[40]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[41]:


movies.head()


# In[42]:


new_df = movies[['movie_id','title','tags']]


# In[43]:


new_df.head()


# In[44]:


new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


# In[45]:


new_df['tags'][0]


# In[46]:


new_df.head()


# In[47]:


new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


# In[48]:


new_df['tags'][0]


# In[49]:


import nltk


# In[50]:


from nltk.stem.porter import PorterStemmer


# In[51]:


ps = PorterStemmer()


# In[52]:


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)    
    


# In[53]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[54]:


new_df.head()


# In[55]:


new_df['tags'][0]


# In[56]:


new_df['tags'][1]


# In[57]:


from sklearn.feature_extraction.text import CountVectorizer


# In[58]:


cv = CountVectorizer(max_features=5000,stop_words='english')


# In[59]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[60]:


vectors


# In[61]:


vectors.shape


# In[62]:


vectors[0]         # avatar movie's vector


# In[63]:


cv.get_feature_names()


# In[64]:


from sklearn.metrics.pairwise import cosine_similarity


# In[65]:


similarity = cosine_similarity(vectors)


# In[66]:


similarity.shape


# In[67]:


similarity[1]


# In[68]:


new_df[new_df['title'] == "Interstellar"].index[0]


# In[69]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])


# In[70]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[71]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distance = similarity[movie_index]
    movie_list = sorted(list(enumerate(distance)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movie_list:
        print(new_df.iloc[i[0]].title)


# In[72]:


new_df.iloc[1216].title


# In[73]:


recommend('X-Men')


# In[74]:


import pickle


# In[75]:


pickle.dump(new_df.to_dict(),open('Movie_dict.pkl','wb'))


# In[76]:


pickle.dump(similarity , open('similarity.pkl','wb'))


# In[77]:


new_df


# In[78]:


movies.head()

