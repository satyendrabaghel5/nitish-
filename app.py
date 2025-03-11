# import gradio as gr
# import pandas as pd
# import ollama
# import json
# import matplotlib.pyplot as plt

# # Define LLM Model Configuration
# MODEL_NAME = "llama2:7b"

# # Function to process CSV file
# def process_csv(file):
#     """Handles CSV file uploads."""
#     try:
#         df = pd.read_csv(file)  # Use file path directly
#         return df
#     except Exception as e:
#         return str(e)

# # Function to query CSV data using Ollama LLM
# def query_csv(file, question):
#     """Processes user questions on the uploaded CSV data using Ollama."""
#     df = process_csv(file)
#     if isinstance(df, str):  # Error handling
#         return df
    
#     prompt = f"""
#     Given the following CSV data:
#     {df.head(5).to_json()}
#     Answer the following question: {question}
#     """
#     response = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
#     return response["message"]["content"]

# # Function to generate graph
# def plot_graph(file, x_col, y_col):
#     """Generates graphs based on the CSV data."""
#     df = process_csv(file)
#     if isinstance(df, str):
#         return df
    
#     # Debugging: Print available columns
#     print("Available columns:", df.columns.tolist())
    
#     # Check if the selected columns exist
#     if x_col not in df.columns or y_col not in df.columns:
#         return f"Error: Columns '{x_col}' or '{y_col}' not found in dataset."
    
#     # Convert columns to numeric to avoid errors
#     df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
#     df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
#     df = df.dropna()  # Remove invalid rows
    
#     plt.figure(figsize=(8, 5))
#     plt.scatter(df[x_col], df[y_col], alpha=0.5)
#     plt.xlabel(x_col)
#     plt.ylabel(y_col)
#     plt.title(f"{x_col} vs {y_col}")
#     plt.grid()
#     plt.savefig("plot.png")
    
#     return "plot.png"

# # Define Gradio UI
# def gradio_app():
#     with gr.Blocks() as demo:
#         gr.Markdown("### CSV Question Answering & Visualization App")
#         file_input = gr.File(label="Upload CSV File", type="filepath")
        
#         with gr.Row():
#             question_input = gr.Textbox(label="Ask a Question")
#             answer_output = gr.Textbox(label="Answer", interactive=False)
#             ask_button = gr.Button("Get Answer")
        
#         ask_button.click(fn=query_csv, inputs=[file_input, question_input], outputs=answer_output)
        
#         with gr.Row():
#             x_column = gr.Textbox(label="X-axis Column Name")
#             y_column = gr.Textbox(label="Y-axis Column Name")
#             plot_output = gr.Image()
#             plot_button = gr.Button("Generate Graph")
        
#         plot_button.click(fn=plot_graph, inputs=[file_input, x_column, y_column], outputs=plot_output)
        
#     return demo

# # Run Gradio app
# demo = gradio_app()
# demo.launch()


import gradio as gr
import pandas as pd
import ollama
import json
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define LLM Model Configuration
MODEL_NAME = "llama2:7b"

# Function to process CSV file
def process_csv(file):
    """Handles CSV file uploads."""
    try:
        df = pd.read_csv(file)  # Use file path directly
        logging.info("CSV file loaded successfully.")
        return df
    except Exception as e:
        logging.error("Error loading CSV file: %s", str(e))
        return f"Error loading CSV file: {str(e)}"

# Function to query CSV data using Ollama LLM
def query_csv(file, question):
    """Processes user questions on the uploaded CSV data using Ollama."""
    df = process_csv(file)
    if isinstance(df, str):  # Error handling
        return df
    
    prompt = f"""
    Given the following CSV data:
    {df.head(5).to_json()}
    Answer the following question: {question}
    """
    try:
        response = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]
    except Exception as e:
        logging.error("Error processing LLM query: %s", str(e))
        return "An error occurred while processing your query. Please try again."

# Function to generate graph
def plot_graph(file, x_col, y_col):
    """Generates graphs based on the CSV data."""
    df = process_csv(file)
    if isinstance(df, str):
        return df
    
    # Log available columns
    logging.info("Available columns: %s", df.columns.tolist())
    
    # Check if the selected columns exist
    if x_col not in df.columns or y_col not in df.columns:
        return f"Error: Columns '{x_col}' or '{y_col}' not found. Available columns: {df.columns.tolist()}"
    
    # Convert columns to numeric to avoid errors
    df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
    df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
    df = df.dropna()  # Remove invalid rows
    
    # Check if valid data exists for plotting
    if df.empty:
        return "Error: No valid data available for plotting after conversion."
    
    plt.figure(figsize=(8, 5))
    plt.scatter(df[x_col], df[y_col], alpha=0.5)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{x_col} vs {y_col}")
    plt.grid()
    plt.savefig("plot.png")
    logging.info("Graph successfully generated.")
    
    return "plot.png"

# Define Gradio UI
def gradio_app():
    with gr.Blocks() as demo:
        gr.Markdown("### CSV Question Answering & Visualization App")
        file_input = gr.File(label="Upload CSV File", type="filepath")
        
        with gr.Row():
            question_input = gr.Textbox(label="Ask a Question")
            answer_output = gr.Textbox(label="Answer", interactive=False)
            ask_button = gr.Button("Get Answer")
        
        ask_button.click(fn=query_csv, inputs=[file_input, question_input], outputs=answer_output)
        
        with gr.Row():
            x_column = gr.Textbox(label="X-axis Column Name")
            y_column = gr.Textbox(label="Y-axis Column Name")
            plot_output = gr.Image()
            plot_button = gr.Button("Generate Graph")
        
        plot_button.click(fn=plot_graph, inputs=[file_input, x_column, y_column], outputs=plot_output)
        
    return demo

# Run Gradio app
demo = gradio_app()
demo.launch()
