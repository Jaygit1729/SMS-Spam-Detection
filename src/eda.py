import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from logger_utils import setup_logger
from data_ingestions import load_data

# Initialize Logger for EDA
logger = setup_logger("logs/eda.log")
logger.info("Logging setup for EDA successfully.")

def describe_messages(df, target_class):
    
    """Returns descriptive statistics for a given class (spam/ham)."""
    
    return df[df['Target'] == target_class][["num_character", "num_words", "num_sentences"]].describe()

def spam_ham_dist(df):
    
    """Generates a bar chart for spam-ham class distribution with percentage labels."""
    
    class_counts = df['Target'].value_counts().reset_index()
    class_counts.columns = ['Target', 'Count']
    
    # Compute percentage distribution
    class_counts['Percentage'] = (class_counts['Count'] / df.shape[0]) * 100
    class_counts['Percentage'] = class_counts['Percentage'].round(2).astype(str) + "%"  # Format percentage
    
    # Create bar chart with percentage as text
    fig = px.bar(data_frame=class_counts, x='Target', y='Count', text='Percentage')
    fig.update_layout(title="Spam-Ham Class Distribution")
    return fig  

def kde_dist(df, target_class):
    
    """Generates KDE approximation plots for a given target class (spam/ham)."""
    
    target_df = df[df['Target'] == target_class]
    features = ['num_character', 'num_words', 'num_sentences']
    fig = make_subplots(rows=1, cols=3, subplot_titles=features)
    
    for i, feature in enumerate(features):
        hist_data = [target_df[feature]]
        group_labels = [feature]
        kde_fig = ff.create_distplot(hist_data, group_labels, show_hist=True, show_rug=False)
        for trace in kde_fig['data']:
            fig.add_trace(trace, row=1, col=i+1)
    
    fig.update_layout(title_text=f"KDE Approximation for {target_class.capitalize()} Messages", 
                      showlegend=False, width=1200, height=400)
    return fig

def pairwise_dist(df):
    
    """Generates pairwise scatter matrix plot."""
    
    fig = px.scatter_matrix(df,
                            dimensions=["num_character", "num_words", "num_sentences"],
                            color="Target",  
                            title="Pairwise Scatter Matrix of Features"
                           )
    fig.update_layout(width=1000, height=800)
    return fig


def plot_correlation_heatmap(df):
    
    """Generates a heatmap for the correlation matrix of the DataFrame using Plotly."""
    
    corr_matrix = df.corr()  
    fig = px.imshow(corr_matrix, 
                    color_continuous_scale='RdBu_r', 
                    title="Correlation Heatmap",
                    labels=dict(x="Features", y="Features"))  
    
    fig.update_layout(width=800, height=600)  
    return fig


if __name__ == "__main__":
    
    df = load_data(r"data/feature_engineering/spam.csv")
    if df is not None:
        logger.info("Loaded cleaned data for EDA.")

        # Generate and show the plot
        fig = spam_ham_dist(df)
        fig.show() 
        
        fig = kde_dist(df, 'spam')
        fig.show()

        fig = kde_dist(df, 'ham')
        fig.show()

        fig = pairwise_dist(df)
        fig.show()

        fig = plot_correlation_heatmap(df)
        fig.show()
