#app.py will be containing the front end interface of the project.With the use of modules of such as streamlit,pandas and intergrations with files like chatbot_logic will be established here.The heavy lifting for the frontend design will be done by the streamlit module.
import streamlit as st
import pandas as pd
from chatbot_logic import handle_query
from utils import get_plot_function_and_cols # For direct plot handling

# --- Page Configuration ---
st.set_page_config(page_title="📊 Data Analytics Chatbot", layout="wide", initial_sidebar_state="collapsed")

# --- Title and Description ---
st.title("📊 Data Analytics Chatbot 🤖")
st.markdown("""
Welcome! Upload your CSV dataset, and then ask me questions about it. I can help with:
* EDA: Show head, summary, column stats, data types, missing values, correlations.
* Visualizations: Histograms, box plots, scatter plots, correlation heatmaps.
* Machine Learning: Train simple regression or classification models.
* General Questions: Ask anything else, and I'll try to answer with the help of Gemini!s
""")

# --- API Key Check (Optional but good practice for local dev) ---
# import os
# if not os.getenv("GOOGLE_API_KEY"):
#     st.warning("⚠ GOOGLE_API_KEY environment variable not set. Gemini features may not work.")

# --- File Uploader ---
with st.sidebar:
    st.header("📁 Dataset Upload")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    st.markdown("---")
    st.markdown("""
    Example Questions:
    - Show the first 5 rows
    - Give me a summary of the data
    - What are the data types?
    - Plot a histogram of 'age'
    - Show a scatter plot of 'salary' vs 'experience'
    - Train a model to predict 'target_variable'
    """)

# --- Main Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Please upload a CSV file using the sidebar to get started."}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], pd.DataFrame):
            st.dataframe(message["content"])
        elif isinstance(message["content"], dict) and message["content"].get("type") == "plotly":
            st.plotly_chart(message["content"]["fig"], use_container_width=True)
        else:
            st.markdown(message["content"])

if uploaded_file:
    if "df" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
        encodings_to_try = ['latin-1', 'windows-1252', 'utf-16', 'cp1250', 'gbk', 'big5']

        for encoding in encodings_to_try:
            try:
                st.session_state.df = pd.read_csv(uploaded_file,encoding=encoding)
                st.session_state.file_name = uploaded_file.name
                st.session_state.messages = [{"role": "assistant", "content": f"Successfully loaded {uploaded_file.name}. What would you like to know?"}]
                with st.expander("🔍 View Uploaded DataFrame (First 100 rows)", expanded=False):
                    st.dataframe(st.session_state.df.head(100))
            # Automatically trigger a rerun to update the chat with the new message
                st.rerun()
            except Exception as e:
                st.error(f"Error loading CSV: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I couldn't load the file. Error: {e}"})
                uploaded_file = None # Reset to prevent further processing with bad file

    if "df" in st.session_state:
        if prompt := st.chat_input("Ask your data-related question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("🧠 Thinking..."):
                    response_data = handle_query(prompt, st.session_state.df)

                    if isinstance(response_data, str):
                        st.markdown(response_data)
                        st.session_state.messages.append({"role": "assistant", "content": response_data})
                    elif isinstance(response_data, pd.DataFrame):
                        st.dataframe(response_data)
                        st.session_state.messages.append({"role": "assistant", "content": response_data})
                    elif isinstance(response_data, dict) and response_data.get("type") == "plotly":
                        st.plotly_chart(response_data["fig"], use_container_width=True)
                        st.session_state.messages.append({"role": "assistant", "content": response_data})
                    elif response_data is None: # Handle cases where a plot might be displayed directly by a util
                        st.markdown("✅ Plot displayed above.") # Or fetch the message from the util if it returns one
                        st.session_state.messages.append({"role": "assistant", "content": "✅ Plot displayed."})
                    else:
                        st.markdown("Sorry, I received an unexpected response type.")
                        st.session_state.messages.append({"role": "assistant", "content": "Sorry, I received an unexpected response type."})
else:
    if not any(msg["content"] == "Hello! Please upload a CSV file using the sidebar to get started." for msg in st.session_state.messages):
        st.session_state.messages.append({"role": "assistant", "content": "Hello! Please upload a CSV file using the sidebar to get started."})

# To ensure the file uploader message appears correctly if no file is uploaded yet.
if not uploaded_file and len(st.session_state.messages) == 1 and st.session_state.messages[0]["content"] == "Hello! Please upload a CSV file using the sidebar to get started.":
    pass # Initial state is fine
elif not uploaded_file and "df" not in st.session_state:
     # If file was previously loaded and then removed or an error occurred
    if not st.session_state.messages or st.session_state.messages[-1]["content"] != "Please upload a CSV file to continue analyzing.":
        st.info("Please upload a CSV file to continue analyzing.")
        # st.session_state.messages.append({"role": "assistant", "content": "Please upload a CSV file to continue analyzing."})