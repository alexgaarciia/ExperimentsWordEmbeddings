# ExperimentsWordEmbeddings

## Subject: COMP472 Artificial Intelligence
#### Teacher: Leila Kosseim

## Team Members
- Alejandro Leonardo García Navarro - [alexgaarciia](https://github.com/alexgaarciia)
- Lucía Cordero Sánchez - [lucia-corsan](https://github.com/lucia-corsan)
- Simon Dunand - [SquintyG33Rs](https://github.com/SquintyG33Rs)

## URL to the repository (private)
- [ExperimentsWordEmbeddings](https://github.com/alexgaarciia/ExperimentsWordEmbeddings).
  
## Languages and software
- Language used; Python.
- Done in Pycharm, but works for any other Python IDE.
  
# Project A2 - Word Embeddings
- Dataset: synonym.csv

- Main tasks:
0. Contribution to a Collective Human Gold-Standard.
1. Evaluation of the word2vec-google-news-300 pre-trained model.
2. Comparison with other pre-trained models.
3. Train our own models.

- Important files:
1. A2-Dataset: Folder that contains the dataset.
2. Books: Folder that contains all of the books needed to train our own models.
3. Details: Here we can find the details of every model. For each question in the synonym test dataset, it has a single line indicating (a) the question word, (b) the correct answer, (c) our system's guess-word, and (d) a label (guess, correct or wrong).
4. COMP472-A2.ipynb: This notebook is divided into several parts (Import necessary libraries/modules, Functions (this part contains the functions that we created to perform the project correctly and in an organized manner (def model_evaluator(model, dataset), def output_df(output_data, model_name), def compute_accuracy(output_df), def save_data(analysis_dic, model_name, model, output_data), and def preprocess_books(books)), and Tasks).
5. analysis.csv: Here we can find the analysis of every model, showing (a) the model name, (b) the size of the vocabulary, (c) the number of correct labels, (d) the number of questions that our model answered without guession, and (e) the accuracy of the model.

- **Instructions**: Run the "COMP472-A2.ipynb" file to perform all of the tasks. Make sure you have the dataset in a folder called "A2-Dataset" and all of the books in a folder called "Books".
