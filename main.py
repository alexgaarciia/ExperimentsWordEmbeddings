# Import necessary libraries/modules
import gensim.downloader
import pandas as pd

# ----------------------------------------------------------------------------------------------------------------------
# TASK 1
# ----------------------------------------------------------------------------------------------------------------------
# Load the pretrained embedding model:
model = gensim.downloader.load("word2vec-google-news-300")

# Load the Synonym Test dataset:
dataset = pd.read_csv("A2-DataSet/synonym.csv")
print(dataset)

# Prepare the output data:
output_data = []

# Iterate through each row in the dataset
for index, row in dataset.iterrows():
    # Extract the question and answer words from the current row
    question_word = row['question']
    answer_word = row['answer']

    # Generate a list of guess words from the current row
    guess_words = [row[str(i)] for i in range(4)]

    # Check if the question word is in the model
    if question_word in model:
        # Initialize a list to store similarities between question and guess words
        similarities = []

        # Loop through each guess word
        for guess_word in guess_words:
            # Check if the guess word is in the model
            if guess_word in model:
                # Calculate the similarity between question and guess word using the similarity method from Gensim
                similarity = model.similarity(question_word, guess_word)
                # Store the guess word and its similarity score
                similarities.append((guess_word, similarity))

        # Determine the best guess based on the highest similarity score
        if similarities:
            best_guess, _ = max(similarities, key=lambda x: x[1])
            # Label as "correct" if the best guess matches the answer word
            label = 'correct' if best_guess == answer_word else 'wrong'
        else:
            # If no valid guesses, set best guess to None and label as "guess"
            best_guess = None
            label = 'guess'
    else:
        # If question word is not in the model, no guess can be made
        best_guess = None
        label = 'guess'

    # Append the results to the output data
    output_data.append([question_word, answer_word, best_guess, label])

# Convert the output data to a DataFrame and save the output to a CSV file:
output_df = pd.DataFrame(output_data, columns=['question-word', 'answer-word', 'guess-word', 'label'])
output_df.to_csv('word2vec-google-news-300-details.csv', index=False)

# In order to create the "analysis.csv" file, some steps must be followed:
# 1. Count the number of correct (C) and non-guess labels (V):
C = (output_df['label'] == 'correct').sum()
V = (output_df['label'] != 'guess').sum()

# 2. Calculate the accuracy:
if V:
    accuracy = C/V
else:
    accuracy = 0

# 3. Create a DataFrame and save to csv:
analysis_df = pd.DataFrame({
    'model_name': ['word2vec-google-news-300'],
    'vocabulary_size': [len(model.key_to_index)],
    'correct_labels': [C],
    'answered_questions': [V],
    'accuracy': [accuracy]
})
analysis_df.to_csv('analysis.csv', index=False)
