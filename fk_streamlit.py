import pandas as pd
import re
import cohere
import streamlit as st

# Load the CSV file
file_path = 'cl_fk.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Initialize Cohere (Replace with your Cohere API key)
cohere_api_key = "I5WzAkJerwsBeK79CZA5zLxBBBVCeJ8o8ADU4XZ6"  # Replace with your actual API key
co = cohere.Client(cohere_api_key)

# Function to recommend products based on user query
def recommend_products(query, df, top_n=5):
    try:
        response = co.generate(
            model='command-xlarge',
            prompt=f"Extract key features or keywords from: {query}",
            max_tokens=50
        )
        # Process generated keywords
        generated_keywords = response.generations[0].text.strip().split()
        escaped_keywords = [re.escape(keyword) for keyword in generated_keywords]
        pattern = '|'.join(escaped_keywords)

        # Filter recommendations
        if 'Cleaned_Review' in df.columns:
            recommendations = df[df['Cleaned_Review'].str.contains(pattern, case=False, na=False)]
            recommendations = recommendations.sort_values(by='TextBlob_Sentiment_Score', ascending=False)
            
            # Remove duplicates based on Product ID, Product Name, and Price
            recommendations = recommendations[['Product ID', 'Product Name', 'Price']].drop_duplicates()
            
            return recommendations.head(top_n)
        else:
            st.error("Cleaned_Review column not found in the dataset.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred while generating recommendations: {e}")
        return pd.DataFrame()

# Streamlit UI Styling
st.markdown(
    """
    <style>
    .main-container {
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
    }
    .left-image {
        margin-right: 20px;
    }
    .main-title {
        font-size: 48px;
        font-weight: bold;
        color: #2E8B57;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Image and Title Layout
st.markdown(
    """
    <div class="main-container">
        <img src="https://images.pexels.com/photos/3756879/pexels-photo-3756879.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1" alt="Phone Image" class="left-image" width="200">
        <div class="main-title">Phone Recommendation Engine</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("Enter your preferences to find the best phones!")

# Input text box for user query
user_query = st.text_input("Describe your ideal phone (e.g., 'best camera', 'long battery life', 'budget-friendly'): ")

if st.button("Recommend"):
    if user_query.strip():
        recommendations = recommend_products(user_query, df)
        if not recommendations.empty:
            st.markdown("### Recommended Phones:")
            st.dataframe(recommendations.reset_index(drop=True))
        else:
            st.warning("No matching phones found. Try a different query!")
    else:
        st.error("Please enter a query to get recommendations.")
