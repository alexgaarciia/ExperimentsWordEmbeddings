# Import necessary libraries/functions/modules
import gensim.downloader
import pandas as pd
from functions import model_evaluator, output_df, compute_accuracy, save_data, preprocess_books

# ----------------------------------------------------------------------------------------------------------------------
# TASK 1
# ----------------------------------------------------------------------------------------------------------------------
# Load the pretrained embedding model:
model_name = "word2vec-google-news-300"
model = gensim.downloader.load(model_name)

# Load the Synonym Test dataset:
dataset = pd.read_csv("A2-DataSet/synonym.csv")

# Save the data:
analysis_dic = {}
save_data(analysis_dic, model_name, model, model_evaluator(model, dataset))

# Convert dictionary to DataFrame:
analysis_df = pd.DataFrame.from_dict(analysis_dic, orient="index")

# Save the DataFrame to a CSV file:
analysis_df.to_csv('analysis.csv', index=False, header=False)


# ----------------------------------------------------------------------------------------------------------------------
# TASK 2
# ----------------------------------------------------------------------------------------------------------------------
# Load the pretrained embedding model:
model_name1 = "glove-wiki-gigaword-300"
model_name2 = "word2vec-ruscorpora-300"
model_name3 = "glove-twitter-100"
model_name4 = "glove-twitter-200"

model1 = gensim.downloader.load(model_name1)
model2 = gensim.downloader.load(model_name2)
model3 = gensim.downloader.load(model_name3)
model4 = gensim.downloader.load(model_name4)

# Save the data:
save_data(analysis_dic, model_name1, model1, model_evaluator(model1, dataset))
save_data(analysis_dic, model_name2, model2, model_evaluator(model2, dataset))
save_data(analysis_dic, model_name3, model3, model_evaluator(model3, dataset))
save_data(analysis_dic, model_name4, model4, model_evaluator(model4, dataset))

# Convert dictionary to DataFrame:
analysis_df = pd.DataFrame.from_dict(analysis_dic, orient="index")

# Save the DataFrame to a CSV file:
analysis_df.to_csv('analysis.csv', index=False, header=False)

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models import Word2Vec
import pandas as pd

# Download NLTK resources (if not done previously)
# nltk.download('punkt')

# ----------------------------------------------------------------------------------------------------------------------
# TASK 3
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# TASK 3
# ----------------------------------------------------------------------------------------------------------------------

books = ['agnes_grey.txt', 'frankestein.txt', 'jane_eyre.txt', 'little_women.txt', 'mrs_dalloway.txt', 'pride_and_prejudice.txt', 'regiment_of_women.txt', 'wuthering_heights.txt']
window_sizes = [100, 200]
embedding_sizes = [100, 300]

for window_size in window_sizes:
    for embedding_size in embedding_sizes:
        model_name = f'Model_W{window_size}_E{embedding_size}'

        # Preprocess books
        sentences = preprocess_books(books)

        # Train Word2Vec model
        model = Word2Vec(sentences, vector_size = embedding_size, window = window_size, min_count = 1, workers = 4)

        # Save details to CSV
        save_data(analysis_dic, model_name, model.wv, model_evaluator(model.wv, dataset))

        # Convert dictionary to DataFrame:
        analysis_df = pd.DataFrame.from_dict(analysis_dic, orient="index")
        # Save the DataFrame to a CSV file:
        analysis_df.to_csv('analysis.csv', index=False, header=False)