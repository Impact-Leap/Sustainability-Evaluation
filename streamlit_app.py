import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import requests
import json
from streamlit_extras.let_it_rain import rain

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity



# Reading a JSON string from a file
with open('mock_data.json', 'r') as file:
    mock_json = json.load(file)

# Reading a system prompt from a text file
with open('system_prompt.txt', 'r') as file:
    system_prompt = file.read()

# Sidebar for additional options or information
with st.sidebar:
    st.image("earth.png", width=300)
    # For example, links or additional instructions
    api_key = st.text_input("Enter your API key", type="password")
    st.write("We require a GPT-4 Turbo API key, specifically the model gpt-4-1106-preview. Please note that usage may incur charges.")

# def get_tfidf_novelty(problem: str, solution: str):
#     try:
#         # Open the file using a context manager
#         with open('AI EarthHack Dataset.csv', 'r') as file:
#             # Read the file contents
#             documents = pd.read_csv('AI EarthHack Dataset.csv', encoding = 'ISO-8859-1')

#     except FileNotFoundError:
#         print("File not found.")
#         return False

#     # convert columns to string
#     documents['problem'] = documents['problem'].astype(str)
#     documents['solution'] = documents['solution'].astype(str)

#     documents = documents.drop('id', axis=1)
#     documents = documents.apply(lambda row: row['problem'] + " " + row['solution'], axis=1).tolist()

#     # new statement is from user input

#     new_statement = problem + " " + solution

#     # TF-IDF Vectorizer
#     # TfidfVectorizer is equivalent to CountVectorizer followed by TfidfTransformer.
#     vectorizer = TfidfVectorizer()
#     tfidf_matrix = vectorizer.fit_transform(documents)
#     cosine_similarities = cosine_similarity(tfidf_matrix)

#     # Flatten the upper triangle of the similarity matrix and filter out self-similarities (value of 1)
#     upper_triangle = cosine_similarities[np.triu_indices_from(cosine_similarities, k=1)]

#     # # Plotting the distribution of cosine similarities
#     # plt.hist(upper_triangle, bins=50, density=True)
#     # plt.xlabel('Cosine Similarity')
#     # plt.ylabel('Density')
#     # plt.title('Distribution of Cosine Similarities')
#     # plt.show()

#     ## Calculate the tfidf for the new input

#     new_statement_tfidf = vectorizer.transform([new_statement])

#     # Compute cosine similarity between the new statement and all documents in the dataset
#     cosine_similarities = cosine_similarity(new_statement_tfidf, tfidf_matrix)

#     # Heuristic threshold setting
#     mean_similarity = np.mean(upper_triangle)
#     std_dev_similarity = np.std(upper_triangle)
#     threshold = mean_similarity + 2 * std_dev_similarity  # i.e.: mean + 2*std_dev

#     # Find the maximum similarity (you can also use other metrics like average similarity)
#     max_similarity = np.max(cosine_similarities)

#     # Evaluate novelty. Low similarity indicates high novelty with respect to the given dataset
#     if max_similarity < 0.5:
#         return max_similarity, True
#     else:
#         return max_similarity, False

    
def example1():
    rain(
        emoji="🌏",
        font_size=30,
        falling_speed=10,
        animation_length="infinite",
    )

def example2():
    rain(
        emoji="🐧",
        font_size=54,
        falling_speed=3,
        animation_length="infinite",
    )
    
example1()
# example2()

# st.title('🌏 Earth Hack')

# Temporary metrics for sustainability evaluation
metrics = [
    "1_No_Poverty", "2_Zero_Hunger", "3_Good_Health_and_Well-being",
    "4_Quality_Education", "5_Gender_Equality", "6_Clean_Water_and_Sanitation",
    "7_Affordable_and_Clean_Energy", "8_Decent_Work_and_Economic_Growth",
    "9_Industry_Innovation_and_Infrastructure", "10_Reduced_Inequality",
    "11_Sustainable_Cities_and_Communities", "12_Responsible_Consumption_and_Production",
    "13_Climate_Action", "14_Life_Below_Water", "15_Life_on_Land",
    "16_Peace_Justice_and_Strong_Institutions", "17_Partnerships_for_the_Goals"
]    # 不好看 再改改

# Emoji function based on score
def score_to_emoji(score):
    if score <= 4:
        return "🔴"  # Red circle for low scores
    elif 5 <= score <= 6:
        return "🟡"  # Yellow circle for medium scores
    else:
        return "🟢"  # Green circle for high scores


# Function to call GPT API and process the response
def evaluate_idea(problem, solution):
    return mock_json
    # # Replace with your actual API endpoint and API key
    # api_endpoint = "https://api.example.com/gpt-evaluate"
    # headers = {
    #     "Authorization": f"Bearer {api_key}",
    #     "Content-Type": "application/json"
    # }
    # payload = {
    #     "problem": problem,
    #     "solution": solution
    # }
    # try:
    #     response = requests.post(api_endpoint, headers=headers, json=payload)
    #     response.raise_for_status()
    #     return response.json()
    # except requests.RequestException as e:
    #     st.error(f"API request failed: {e}")
    #     return None


# Section for business idea input
st.title("💡 Business Idea Evaluation")
with st.form("business_idea_form"):
    problem = st.text_area("Problem:")
    solution = st.text_area("Solution:")
    submit_button = st.form_submit_button("Evaluate Idea")


# 大头
# if submit_button:
#     if not api_key:
#         st.error("Please enter an API key.")
#     else:
#         api_response = evaluate_idea(problem, solution)
#         if api_response:
#             is_sustainable = api_response['Idea_Sustainability_Related'] != "Yes"
#             if not is_sustainable:
#                 st.write(f"Reason: {api_response['Idea_Sustainability_Related']}")
#             else:
#                 scores = [int(api_response['Evaluation']['SDG_Scores'][metric]['Score']) for metric in metrics]
#                 total_score = int(api_response['Evaluation']['Total_Score'])
#                 analysis_context = api_response['Evaluation']['Summary']
    
#                 color = "green" if total_score > 130 else "red" if total_score < 85 else "yellow"
               
#                 # Display the summary score with color
#                 st.markdown(f"<h3 style='color:{color};'>Summary Score: {total_score:.2f} / 170</h3>", unsafe_allow_html=True)
               
#                 # Displaying summary analysis
#                 st.write("### Summary Analysis:")
#                 st.write(analysis_context)
    
#                 # Create DataFrame for scores and emojis
#                 score_df = pd.DataFrame({
#                     'Metric': metrics,
#                     'Score': scores,
#                     'Level': [score_to_emoji(score) for score in scores]
#                 })
            
#                 # Displaying scores as a styled table using markdown
#                 st.markdown("### Evaluation Table")
#                 st.markdown(
#                     score_df.to_html(index=False, escape=False, justify='center', classes='table'),
#                     unsafe_allow_html=True
#                 )
            
#                 # Apply custom CSS for table styling
#                 st.markdown("""
#                     <style>
#                         .table {width: 100%; margin-left: auto; margin-right: auto; border-collapse: collapse;}
#                         .table td, .table th {border: none;}
#                         th {text-align: center; font-size: 18px; font-weight: bold;}
#                         td {text-align: center;}
#                     </style>
#                     """, unsafe_allow_html=True)
    
#                 # Slider section
#                 st.write("### Evaluation Results:")
#                 for metric, score in zip(metrics, scores):
#                     st.slider(metric, 0, 10, score, disabled=True)
            
#                 # Radar chart
#                 st.write("### Radar Chart Evaluation Results:")
#                 num_vars = len(metrics)
#                 angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
#                 angles += angles[:1]  # Complete the loop
            
#                 scores_list = scores.tolist()
#                 scores_list += scores_list[:1]  # Repeat the first score to close the radar chart
            
#                 fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
#                 ax.fill(angles, scores_list, color='green', alpha=0.25)
#                 ax.plot(angles, scores_list, color='green', linewidth=2)
#                 ax.set_yticklabels([])
#                 ax.set_xticks(angles[:-1])
#                 ax.set_xticklabels(metrics)
            
#                 st.pyplot(fig)
                
#                 # Seaborn barplot
#                 st.write("### Bar Chart Evaluation Results:")
#                 plt.figure(figsize=(10, 6))
#                 sns.barplot(x='Score', y='Metric', data=score_df, palette="vlag")
#                 plt.xlabel('Score out of 10')
#                 st.set_option('deprecation.showPyplotGlobalUse', False)
#                 st.pyplot()
            
#     ## Simulate scores for demonstration (replace with real data later)
#     # scores = np.random.randint(1, 11, size=len(metrics))

#     ## Calculate the summary score
#     # total_score = sum(scores)
#     # normalized_score = total_score * (170 / (10 * len(metrics)))  # Normalizing to a scale of 170   # 之后会改掉哈，直接全加起来

#         else:
#             # Display warning message if API call fails
#             st.error("Unable to retrieve data. Please try again later.")


if submit_button:
        api_response = evaluate_idea(problem, solution)
        if api_response:
            # Display if the idea is sustainability related with highlighting
            # is_sustainable = api_response['Idea_Sustainability_Related'] == "Yes"
            is_sustainable = api_response['Idea_Sustainability_Related'] == True
            sustainability_comment = api_response['Idea_Sustainability_Related_Comment']
            st.markdown(f"<h3 style='color:blue;'>Is the Idea Sustainability Related? {'Yes' if is_sustainable else 'No'}</h3>", unsafe_allow_html=True)
            st.write("### Sustainability Analysis:")
            st.write(sustainability_comment)
    
            if is_sustainable:
                scores = [int(api_response['Evaluation']['SDG_Scores'][metric]['Score']) for metric in metrics]
                comments = [api_response['Evaluation']['SDG_Scores'][metric]['Comment'] for metric in metrics]
                total_score = int(api_response['Evaluation']['Total_Score'])  # Convert to integer
                analysis_context = api_response['Evaluation']['Summary']
                novelty_score = int(api_response['Evaluation']['Novelty_Score'])  # Convert to integer
                novelty_comment = api_response['Evaluation']['Novelty_Evaluation']['Comment']
                
                
                # Displaying novelty score and analysis with highlighting
                st.markdown(f"<h3 style='color:purple;'>Novelty Score: {novelty_score} / 100</h3>", unsafe_allow_html=True)
                st.write("### Novelty Analysis:")
                st.write(novelty_comment)

                
                # # Example usage
                # max_similarity, tfidf_is_novelty = get_tfidf_novelty("how to solve the spam use of plastic?", "Use paper bags instead of plastic bags")
                
                # # Check if the response is novelty or not
                # novelty_status = "Yes" if tfidf_is_novelty else "No"
                
                # # Display the novelty similarity score and novelty status
                # st.markdown(f"<h3 style='color:orange;'>Novelty Similarity Score: {max_similarity:.2f} / 100</h3> <p>Is it Novelty? {novelty_status}</p>", unsafe_allow_html=True)

                
                # Display the summary score without decimals
                st.markdown(f"<h3 style='color:green;'>Summary Score: {total_score} / 100</h3>", unsafe_allow_html=True)
                st.write("### Summary Analysis:")
                st.write(analysis_context)

                
                # Modify DataFrame to include comments
                score_df = pd.DataFrame({
                    'Metric': metrics,
                    'Score': scores,
                    'Comment': comments,
                    'Level': [score_to_emoji(score) for score in scores]
                })
            
                # Displaying scores as a styled table using markdown
                st.markdown("### Evaluation Table")
                st.markdown(
                    score_df.to_html(index=False, escape=False, justify='center', classes='table'),
                    unsafe_allow_html=True
                )
            
                # Apply custom CSS for table styling
                st.markdown("""
                    <style>
                        .table {width: 100%; margin-left: auto; margin-right: auto; border-collapse: collapse;}
                        .table td, .table th {border: none;}
                        th {text-align: center; font-size: 18px; font-weight: bold;}
                        td {text-align: center;}
                    </style>
                    """, unsafe_allow_html=True)
                
                
                # Slider section
                st.write("### Evaluation Results:")
                for metric, score in zip(metrics, scores):
                    st.slider(metric, 0, 10, score, disabled=True)
            
                # Radar chart
                st.write("### Radar Chart Evaluation Results:")
                num_vars = len(metrics)
                angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
                angles += angles[:1]  # Complete the loop
            
                # scores_list = scores.tolist()
                # scores_list += scores_list[:1]  # Repeat the first score to close the radar chart

                # Use 'scores' directly as it is already a list
                scores_list = scores + scores[:1]  # Repeat the first score to close the radar chart

                fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
                ax.fill(angles, scores_list, color='green', alpha=0.25)
                ax.plot(angles, scores_list, color='green', linewidth=2)
                ax.set_yticklabels([])
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(metrics)
            
                st.pyplot(fig)
                
                # Seaborn barplot
                st.write("### Bar Chart Evaluation Results:")
                plt.figure(figsize=(10, 6))
                sns.barplot(x='Score', y='Metric', data=score_df, palette="vlag")
                plt.xlabel('Score out of 10')
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()
            
    ## Simulate scores for demonstration (replace with real data later)
    # scores = np.random.randint(1, 11, size=len(metrics))

    ## Calculate the summary score
    # total_score = sum(scores)
    # normalized_score = total_score * (170 / (10 * len(metrics)))  # Normalizing to a scale of 170   # 之后会改掉哈，直接全加起来

        else:
            # Display warning message if API call fails
            st.error("Unable to retrieve data. Please try again later.")

    # # Determine the color based on the score
    # if normalized_score < 85:
    #     color = "red"
    # elif normalized_score > 130:
    #     color = "green"
    # else:
    #     color = "yellow"

    # # Display the summary score with color
    # st.markdown(f"<h3 style='color:{color};'>Summary Score: {normalized_score:.2f} / 170</h3>", unsafe_allow_html=True)

    # # Placeholder for summary analysis from API
    # st.write("### Summary Analysis:")
    # # Placeholder text - Replace with API call and response handling
    # st.write("Analysis will be displayed here once the API is integrated.")


