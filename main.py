# Import necessary libraries/functions/modules
import gensim.downloader
import pandas as pd
from functions import model_evaluator, output_df, compute_accuracy, save_data

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

