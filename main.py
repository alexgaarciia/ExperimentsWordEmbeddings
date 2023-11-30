# Import necessary libraries/functions/modules
import nltk
import pandas as pd
import gensim.downloader
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize, sent_tokenize
from functions import model_evaluator, output_df, compute_accuracy, save_data, preprocess_books
# nltk.download('punkt')  # Download NLTK resources (if not done previously)


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
model_name2 = "fasttext-wiki-news-subwords-300"
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


# ----------------------------------------------------------------------------------------------------------------------
# TASK 3
# ----------------------------------------------------------------------------------------------------------------------
# Declare variables that will be used in the task, such as the books, the window sizes or the embedding sizes:
books = ['Books/agnes_grey.txt', 'Books/frankestein.txt', 'Books/jane_eyre.txt',
         'Books/little_women.txt', 'Books/mrs_dalloway.txt', 'Books/pride_and_prejudice.txt',
         'Books/regiment_of_women.txt', 'Books/wuthering_heights.txt']
window_sizes = [100, 200]
embedding_sizes = [100, 300]

for window_size in window_sizes:
    for embedding_size in embedding_sizes:
        model_name = f'Model_W{window_size}_E{embedding_size}'

        # Preprocess books:
        sentences = preprocess_books(books)

        # Train Word2Vec model:
        model = Word2Vec(sentences, vector_size = embedding_size, window = window_size, min_count = 1, workers = 4)

        # Save details to CSV:
        save_data(analysis_dic, model_name, model.wv, model_evaluator(model.wv, dataset))

        # Convert dictionary to DataFrame:
        analysis_df = pd.DataFrame.from_dict(analysis_dic, orient="index")

        # Save the DataFrame to a CSV file:
        analysis_df.to_csv('analysis.csv', index=False, header=False)

# ----------------------------------------------------------------------------------------------------------------------
# TASK 3.2: Evaluation of models
# ----------------------------------------------------------------------------------------------------------------------
# Load analysis results from the CSV file:
analysis_df = pd.read_csv('analysis.csv', header=None,
                          names=['model_name', 'vocabulary_size', 'correct_labels', 'answered_questions', 'accuracy'])

# Sort the DataFrame by accuracy in descending order:
analysis_df = analysis_df.sort_values(by='accuracy', ascending=False)

# Plot the bar graph:
plt.figure(figsize=(10, 6))
plt.bar(analysis_df['model_name'], analysis_df['accuracy'], color=['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray'])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Comparison in terms of accuracy')
plt.ylim(0, 1)  # Set y-axis limit to better visualize differences
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()

# Save the plot as an image (optional):
plt.savefig('model_comparison_accuracy.png')

# Display the plot:
plt.show()
