# Import necessary libraries
import pandas as pd


# Function used to evaluate the model:
def model_evaluator(model, dataset):
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
    return output_data


# Function used to transform the data into a DataFrame and CSV file:
def output_df(output_data, model_name):
    # Convert the output data to a DataFrame and save the output to a CSV file:
    output_df = pd.DataFrame(output_data, columns=['question-word', 'answer-word', 'guess-word', 'label'])
    output_df.to_csv(model_name + "-details.csv", index=False)
    return output_df


# Function used to compute the metrics:
def compute_accuracy(output_df):
    # 1. Count the number of correct (C) and non-guess labels (V):
    C = (output_df['label'] == 'correct').sum()
    V = (output_df['label'] != 'guess').sum()

    # 2. Calculate the accuracy:
    if V:
        accuracy = C / V
    else:
        accuracy = 0
    return C, V, accuracy


# Function used to save all the details of several models in a dictionary:
def save_data(analysis_dic, model_name, model, output_data):
    C, V, accuracy = compute_accuracy(output_df(output_data, model_name))
    new_entry = {
        'model_name': model_name,
        'vocabulary_size': len(model.key_to_index),
        'correct_labels': C,
        'answered_questions': V,
        'accuracy': accuracy
    }
    analysis_dic[model_name] = new_entry

# Preprocess function to tokenize and split sentences
def preprocess_books(books):
    sentences = []
    for book in books:
        with open(book, 'r', encoding='utf-8') as file:
            text = file.read()
            # Tokenize into sentences
            # .extend removes duplicates from the vocabulary list
            sentences.extend(sent_tokenize(text))
    return [word_tokenize(sentence.lower()) for sentence in sentences]