import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import requests
import json
from streamlit_extras.let_it_rain import rain

# mock json data

mock_jason = {
  "Idea": "EchoMe Voice-to-Text Software",
  "Idea_Sustainability_Related": "EchoMe aligns with several UN SDGs by enhancing accessibility, promoting inclusive technology, and potentially improving educational and work efficiencies.",
  "Evaluation": {
    "SDG_Scores": {
      "1_No_Poverty": {
        "Score": "3",
        "Comment": "Indirect impact on poverty reduction by potentially enabling new job opportunities for the disabled."
      },
      "2_Zero_Hunger": {
        "Score": "0",
        "Comment": "No direct relevance to hunger and food security."
      },
      "3_Good_Health_and_Well-being": {
        "Score": "7",
        "Comment": "Promotes well-being by enhancing accessibility for individuals with disabilities."
      },
      "4_Quality_Education": {
        "Score": "8",
        "Comment": "Could significantly aid in educational accessibility and efficiency, especially for those with writing or mobility impairments."
      },
      "5_Gender_Equality": {
        "Score": "4",
        "Comment": "Indirectly supports gender equality by providing equal access to technology for both women and men."
      },
      "6_Clean_Water_and_Sanitation": {
        "Score": "0",
        "Comment": "Not applicable to water and sanitation issues."
      },
      "7_Affordable_and_Clean_Energy": {
        "Score": "2",
        "Comment": "Minimal relevance, though energy-efficient software design could be a consideration."
      },
      "8_Decent_Work_and_Economic_Growth": {
        "Score": "6",
        "Comment": "Can enhance productivity and open new employment avenues, particularly for the disabled."
      },
      "9_Industry_Innovation_and_Infrastructure": {
        "Score": "8",
        "Comment": "Directly contributes to innovation and technological infrastructure development."
      },
      "10_Reduced_Inequality": {
        "Score": "9",
        "Comment": "Significant potential in reducing inequality by empowering people with disabilities."
      },
      "11_Sustainable_Cities_and_Communities": {
        "Score": "3",
        "Comment": "Indirect impact on sustainable urban development through enhanced accessibility."
      },
      "12_Responsible_Consumption_and_Production": {
        "Score": "2",
        "Comment": "Minimal direct impact, but responsible software development practices could be aligned with this goal."
      },
      "13_Climate_Action": {
        "Score": "1",
        "Comment": "Negligible direct impact on climate action."
      },
      "14_Life_Below_Water": {
        "Score": "0",
        "Comment": "No relevance to marine life conservation."
      },
      "15_Life_on_Land": {
        "Score": "0",
        "Comment": "No direct relevance to life on land conservation."
      },
      "16_Peace_Justice_and_Strong_Institutions": {
        "Score": "4",
        "Comment": "Could indirectly support more inclusive institutions by enhancing communication for all."
      },
      "17_Partnerships_for_the_Goals": {
        "Score": "5",
        "Comment": "Potential for partnerships in technology and innovation sectors."
      }
    },
    "Total_Score": "62",
    "Summary": "EchoMe, a voice-to-text software, aligns with UN SDGs by enhancing accessibility for the disabled, promoting educational inclusivity, and supporting innovation. It has particularly strong potential for reducing inequalities (SDG 10) and aiding in quality education (SDG 4).",
    "Novelty_Score": "60",
    "Novelty_Evaluation": {
      "Comment": "While voice-to-text technology is not new, EchoMe's emphasis on universal accessibility and specific design features for different user groups is novel. The idea carries moderate novelty with the potential for significant societal impact."
    }
  }
}


# Sidebar for additional options or information
with st.sidebar:
    st.image("earth.png", width=300)
    # For example, links or additional instructions
    api_key = st.text_input("Enter your API key", type="password")
    
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
#             is_sustainable = api_response['Idea_ Sustainability_Related '] != "No"
#             if not is_sustainable:
#                 st.write(f"Reason: {api_response['Idea_ Sustainability_Related ']}")
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
            is_sustainable = api_response['Idea_ Sustainability_Related '] != "No"
            if not is_sustainable:
                st.write(f"Reason: {api_response['Idea_ Sustainability_Related ']}")
            else:
                scores = [int(api_response['Evaluation']['SDG_Scores'][metric]['Score']) for metric in metrics]
                total_score = int(api_response['Evaluation']['Total_Score'])
                analysis_context = api_response['Evaluation']['Summary']
    
                color = "green" if total_score > 130 else "red" if total_score < 85 else "yellow"
               
                # Display the summary score with color
                st.markdown(f"<h3 style='color:{color};'>Summary Score: {total_score:.2f} / 170</h3>", unsafe_allow_html=True)
               
                # Displaying summary analysis
                st.write("### Summary Analysis:")
                st.write(analysis_context)
    
                # Create DataFrame for scores and emojis
                score_df = pd.DataFrame({
                    'Metric': metrics,
                    'Score': scores,
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
            
                scores_list = scores.tolist()
                scores_list += scores_list[:1]  # Repeat the first score to close the radar chart
            
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


