from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import openai

def get_top_5_tfidf(problem: str, solution: str):
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

    # Find the indices of the top 5 most similar documents
    top_5_indices = np.argsort(cosine_similarities)[-5:]

    # Retrieve the top 5 most similar documents (rows) from the original DataFrame
    top_5_similar_docs = documents.iloc[top_5_indices]

    # Calculate the average values for NumCompetitors and TotalRaised
    avg_num_competitors = int(round(top_5_similar_docs['NumCompetitors'].mean()))
    avg_total_raised = round(top_5_similar_docs['TotalRaised'].mean(),2)

    return top_5_similar_docs, avg_num_competitors, avg_total_raised

def get_business_status_distribution(top_5_rows):
    # Define the categories
    categories = ['Sustainability', 'CircularEconomy', 'Random', 'Adversarial']

    # Initialize a list to store the results
    distribution = []

    # Calculate the total number of rows in the top 10
    total_rows = top_5_rows.shape[0]

    # Iterate over each category
    for category in categories:
        # Filter rows where the category is marked (value is 1)
        filtered_rows = top_5_rows[top_5_rows[category] == 1]

        # Count the number of occurrences of each business status in this category
        status_counts = filtered_rows['BusinessStatus'].value_counts()

        # Calculate the percentage for each business status
        for status, count in status_counts.items():
            percentage = (count / total_rows) * 100  # Calculate percentage relative to total_rows
            distribution.append({'BusinessStatus': status, 'Category': category, 'Percentage': percentage})

    # Convert the list of dictionaries to a DataFrame
    distribution_df = pd.DataFrame(distribution)

    return distribution_df

from scipy import stats
def get_percentile_by_category(df, column_name, value, category):
    """
    Returns the percentile of a value in a specified column of a DataFrame,
    filtered by a specific category.

    Parameters:
    df (DataFrame): The DataFrame to search.
    column_name (str): The name of the column.
    value (float/int): The value whose percentile we want to find.
    category (str): The category to filter by.

    Returns:
    float: The percentile of the value in the column, within the given category.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")
    if category not in df.columns:
        raise ValueError(f"Category '{category}' not found in DataFrame.")

    # Filter the DataFrame by the category
    filtered_df = df[df[category] == 1]

    # Extract the column from the filtered DataFrame
    column_data = filtered_df[column_name]

    # Calculate the percentile of the value in the filtered data
    percentile = stats.percentileofscore(column_data, value, kind='rank')

    return percentile

def generate_commercial_analysis(NumCompetitors_percentile, most_likely_category, most_likely_business_status, TotalRaised_percentile, avg_num_competitors, avg_total_raised):
    # summary = "Commercial Analysis Summary\n\n"
    # summary = "Congratulations on developing your innovative idea! After a thorough comparison with our extensive industry database, we've gathered insightful findings for your venture:\n\n"

    summary = f"Category Insight: Your venture aligns closely with the '{most_likely_category}' category.\n\n"
    summary += f"Development Stage: Your project appears to be in the '{most_likely_business_status}' stage.\n\n"


    if TotalRaised_percentile == 100:
        summary += f"Financial Potential: Your idea demonstrates extraordinary potential! Showcasing exceptional funding potential and market readiness.\n\n"
    elif TotalRaised_percentile > 50:
        summary += f"Financial Potential: Your idea shows remarkable promise! With an estimated funding capability of {avg_total_raised} million dollars, it ranks in the top {100 - TotalRaised_percentile:.2f}% of its category in the industry.\n\n"
    else:
        summary += f"Financial Potential: Your idea holds significant value. Although slightly below the industry average, it could potentially raise {avg_total_raised} million dollars. There's substantial room for growth and success!\n\n"

    # if TotalRaised_percentile > 50:
    #     summary += f"Financial Potential: Your idea shows remarkable promise! With an estimated funding capability of {avg_total_raised} million dollars, it ranks in the top {100 - TotalRaised_percentile:.2f}% of its category in the industry.\n\n"
    # else:
    #     summary += f"Financial Potential: Your idea holds significant value. Although slightly below the industry average, it could potentially raise {avg_total_raised} million dollars. There's substantial room for growth and success!\n\n"

    if NumCompetitors_percentile < 20:
        summary += f"Market Competition: With less than 20% competition in this sector, your idea could be a game-changer or even a Unicorn! Expect to face about {avg_num_competitors} competitors, setting the stage for a potential market lead."
    elif 20 <= NumCompetitors_percentile < 50:
        summary += f"Market Competition: This emerging field presents an opportunity with limited competition. You're likely to face around {avg_num_competitors} competitors, offering a chance to become a front-runner."
    else:
        summary += f"Market Competition: Prepare for a challenging yet rewarding journey. The sector is competitive, with an average of {avg_num_competitors} competitors, but your unique approach can still make a significant impact."


    return summary

# Example usage:
# print(generate_commercial_analysis(NumCompetitors_percentile, most_likely_category, most_likely_business_status, TotalRaised_percentile, avg_num_competitors, avg_total_raised))

