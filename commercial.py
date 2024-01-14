from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

def get_top_10_tfidf(problem: str, solution: str):
    try:
        # Read the file contents
        documents = pd.read_csv('Cleaned_ValidationSet.csv', encoding='ISO-8859-1')

    except FileNotFoundError:
        print("File not found.")
        return

    # Combine problem and solution into a single string for each row
    documents['Problem_Solution_Pair'] = documents['Problem'] + " " + documents['Solution']

    # Create a list of all problem-solution pairs
    doc_list = documents['Problem_Solution_Pair'].tolist()

    # Calculate TF-IDF for the documents
    vectorizer = TfidfVectorizer()
    doc_tfidf_matrix = vectorizer.fit_transform(doc_list)

    # Add the user input to this list and calculate its TF-IDF
    user_input = problem + " " + solution
    user_input_tfidf = vectorizer.transform([user_input])

    # Compute cosine similarity
    cosine_similarities = cosine_similarity(user_input_tfidf, doc_tfidf_matrix).flatten()

    # Find the indices of the top 10 most similar documents
    top_10_indices = np.argsort(cosine_similarities)[-10:]

    # Retrieve the top 10 most similar documents (rows) from the original DataFrame
    top_10_similar_docs = documents.iloc[top_10_indices]

    # Calculate the average values for NumCompetitors and TotalRaised
    avg_num_competitors = int(round(top_10_similar_docs['NumCompetitors'].mean()))
    avg_total_raised = round(top_10_similar_docs['TotalRaised'].mean(),2)

    return top_10_similar_docs, avg_num_competitors, avg_total_raised


def get_business_status_distribution(top_10_rows):
    # Define the categories
    categories = ['Sustainability', 'CircularEconomy', 'Random', 'Adversarial']

    # Initialize a list to store the results
    distribution = []

    # Calculate the total number of rows in the top 10
    total_rows = top_10_rows.shape[0]

    # Iterate over each category
    for category in categories:
        # Filter rows where the category is marked (value is 1)
        filtered_rows = top_10_rows[top_10_rows[category] == 1]

        # Count the number of occurrences of each business status in this category
        status_counts = filtered_rows['BusinessStatus'].value_counts()

        # Calculate the percentage for each business status
        for status, count in status_counts.items():
            percentage = (count / total_rows) * 100  # Calculate percentage relative to total_rows
            distribution.append({'BusinessStatus': status, 'Category': category, 'Percentage': percentage})

    # Convert the list of dictionaries to a DataFrame
    distribution_df = pd.DataFrame(distribution)

    return distribution_df
