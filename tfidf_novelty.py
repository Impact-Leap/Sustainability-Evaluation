# util program, given a dataset that contains circular economy business ideas that come in problem-solution pairs,
# this program is able to take a problem and solution pair as input and compare it with the dataset to calculate the
# novelty score. Unseen idea will be marked as novel by this function.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# parameters: takes a string of problem and a string of solution
# return: similarity score and truth of novelty
def get_tfidf_novelty(problem: str, solution: str):
    try:
        # Open the file using a context manager
        with open('AI EarthHack Dataset.csv', 'r') as file:
            # Read the file contents
            documents = pd.read_csv('AI EarthHack Dataset.csv', encoding = 'ISO-8859-1')

    except FileNotFoundError:
        print("File not found.")
        return

    # convert columns to string
    documents['problem'] = documents['problem'].astype(str)
    documents['solution'] = documents['solution'].astype(str)

    documents = documents.drop('id', axis=1)
    documents = documents.apply(lambda row: row['problem'] + " " + row['solution'], axis=1).tolist()

    # new statement is from user input

    new_statement = problem + " " + solution

    # TF-IDF Vectorizer
    # TfidfVectorizer is equivalent to CountVectorizer followed by TfidfTransformer.
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    cosine_similarities = cosine_similarity(tfidf_matrix)

    # Flatten the upper triangle of the similarity matrix and filter out self-similarities (value of 1)
    upper_triangle = cosine_similarities[np.triu_indices_from(cosine_similarities, k=1)]

    # # Plotting the distribution of cosine similarities
    # plt.hist(upper_triangle, bins=50, density=True)
    # plt.xlabel('Cosine Similarity')
    # plt.ylabel('Density')
    # plt.title('Distribution of Cosine Similarities')
    # plt.show()

    ## Calculate the tfidf for the new input

    new_statement_tfidf = vectorizer.transform([new_statement])

    # Compute cosine similarity between the new statement and all documents in the dataset
    cosine_similarities = cosine_similarity(new_statement_tfidf, tfidf_matrix)

    # Heuristic threshold setting
    mean_similarity = np.mean(upper_triangle)
    std_dev_similarity = np.std(upper_triangle)
    threshold = mean_similarity + 2 * std_dev_similarity  # i.e.: mean + 2*std_dev

    # Find the maximum similarity (you can also use other metrics like average similarity)
    max_similarity = np.max(cosine_similarities)

    # Evaluate novelty. Low similarity indicates high novelty with respect to the given dataset
    if max_similarity < 0.5:
        return max_similarity, True
    else:
        return max_similarity, False

# example usage
# print(get_tfidf_novelty("how to solve the spam use of plastic?", "Use paper bags instead of plastic bags"))
