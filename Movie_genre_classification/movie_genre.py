import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
#genre list
genre_list = ['action','adventure','comedy','biography','family','fantasy','documentry','adult','animation','crime','game-show','history','mystery','music','musical','horror','news','reality-tv-show','romance','sci-fi','short-film','sport','talk-show','thriller','war','western']
#if model cannot predict the genre
Unknown_genre='Unknown'

#loading training dataset
try:
    with tqdm(total=50, desc="Loading Training Data") as trdata:
        train_data = pd.read_csv('train_data.txt', sep=':::', header=None, names=['Serialnumber','MOVIE_NAME','Genre','MOVIE_PLOT'], engine='python')
        trdata.update(50)
except Exception as e:
    print(f"Error while loading train_data: {e}")
    raise

#data cleaning and preprocessing for training data
print(train_data.columns)
x_ax_train= train_data['MOVIE_PLOT'].astype(str).apply(lambda doc: doc.lower())
genre_labels=[genre.split(', ') for genre in train_data['Genre']]
mlb= MultiLabelBinarizer()
y_ax_train= mlb.fit_transform(genre_labels)

#vectorizer
tfidf_vect= TfidfVectorizer(max_features=5000) #adjustable

#transforming progress bar
with tqdm(total=50,desc="Vectorizing Training data") as trdata:
    x_train_tfidf = tfidf_vect.fit_transform(x_ax_train)
    trdata.update(50)

#NB Classifier with training data
    with tqdm(total=50, desc="Training Model") as trdata:
        naive_bayes= MultinomialNB()
        multi_output_classifier = MultiOutputClassifier(naive_bayes)
        multi_output_classifier.fit(x_train_tfidf, y_ax_train)
        trdata.update(50)
#load test data
try:
    with tqdm(total=50, desc="Loading test data") as trdata:
        test_data= pd.read_csv('test_data.txt',sep=':::', header=None, names=['SerialNumber', 'MOVIE_NAME','MOVIE_PLOT'],engine='python')
        trdata.update(50)
except Exception as e:
    print(f"Error loading test data: {e}")
    raise

#Data preprocessing for test data
x_ax_test = test_data['MOVIE_PLOT'].astype(str).apply(lambda doc: doc.lower())
 
#Progress bar for test data
with tqdm(total=50, desc="Vectorizing Test Data") as trdata:
    x_test_tfidf = tfidf_vect.transform(x_ax_test)
    trdata.update(50)

#predict genres on the test data
with tqdm(total=50 , desc="Predicting test data") as trdata:
    y_ax_pred = multi_output_classifier.predict(x_test_tfidf)
    trdata.update(50)

#Dataframe for test data with movie name and genres
test_movie_names= test_data['MOVIE_NAME']
PREDICTED_GENRES= mlb.inverse_transform(y_ax_pred)
test_results = pd.DataFrame({'MOVIE_NAME': test_movie_names, 'PREDICTED_GENRES' : PREDICTED_GENRES})

#Retest and replace unprdicted genres
test_results['PREDICTED_GENRES']= test_results['PREDICTED_GENRES'].apply(lambda genres: [Unknown_genre] if len(genres) == 0 else genres)

#write output file with crt format
with open("Model_evaluation.txt", "w", encoding="utf-8") as output_file:
    for _,row in test_results.iterrows():
        movie_name = row['MOVIE_NAME']
        genre_str = ', '.join(row['PREDICTED_GENRES'])
        output_file.write(f"{movie_name} ::: {genre_str}\n")

#Evaluation metrics using training labels
y_train_predict =multi_output_classifier.predict(x_train_tfidf)

#metrics
accuracy =accuracy_score(y_ax_train, y_train_predict)
precision = precision_score(y_ax_train, y_train_predict, average='micro')
recall = recall_score(y_ax_train, y_train_predict,average='micro')
f1 = f1_score(y_ax_train, y_train_predict, average='micro')

#write metriics to output file
with open("Model_evaluation.txt", "a", encoding="utf-8") as output_file:
    output_file.write("\n\nModel Evaluation Metrics:\n")
    output_file.write(f"Accuracy: {accuracy * 100:.2f}%\n ")
    output_file.write(f"Precision: {precision:.2f}%\n")
    output_file.write(f"Recall: {recall:.2f}\n")
    output_file.write(f"F1-Score: {f1:.2f}\n")

print("Model evaluation results and metrics have been saved to 'model_evaluation.txt.")


     



        
        
        