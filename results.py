import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate

# Function to load the movie genre dataset
def load_movie_data():
    return pd.read_csv('data/movie_genre.csv')  # Make sure your CSV contains 'movie_id', 'movie_name', 'genre'

# Function to recommend movies based on RMSE
def get_best_movie_recommendation_by_rmse(metrics_df, movie_genre_df, top_n=5):
    if 'rmse' not in metrics_df.columns:
        raise KeyError("'rmse' column not found in the dataset.")
    
    metrics_df['movie_id'] = metrics_df.index + 1  # Create a new movie_id column
    merged_df = metrics_df.merge(movie_genre_df[['movie_id', 'movie_name']], on='movie_id', how='left')
    
    sorted_metrics = merged_df.sort_values(by='rmse', ascending=True)
    top_recommendations = sorted_metrics.head(top_n)
    
    return top_recommendations[['movie_name', 'rmse']]

# Function to load metrics results
def get_metrics(path='data/results/metrics'):
    results = []
    for f in os.listdir(path):
        if f.startswith('.'):
            continue
        if f.endswith('.csv'):
            model, dataset = f[:-4].split('_')
            df = pd.read_csv(f'{path}/{f}')
            df['model'] = model
            df['dataset'] = dataset
            results.append(df)
    results = pd.concat(results)
    
    if 'rmse' in results.columns:
        return results[['rmse']]  # Only return RMSE data, assuming the movies can be inferred from the index
    else:
        raise KeyError("'rmse' column not found in the dataset.")

# Function to get top movies by genre
def get_top_movies_by_genre(movie_name, movie_genre_df, metrics_df, top_n=5):
    movie_name = movie_name.strip().lower()
    movie_genre = movie_genre_df[movie_genre_df['movie_name'].str.lower() == movie_name]['genre'].values
    
    if len(movie_genre) == 0:
        raise ValueError(f"Movie '{movie_name}' not found in the dataset.")
    
    movie_genre = movie_genre[0]
    genre_movies_df = movie_genre_df[movie_genre_df['genre'].str.contains(movie_genre, case=False, na=False)]
    genre_movies_with_metrics = genre_movies_df.merge(metrics_df[['rmse']], left_index=True, right_index=True, how='left')
    
    genre_movies_sorted = genre_movies_with_metrics.sort_values(by='rmse', ascending=True)
    top_movies = genre_movies_sorted.head(top_n)
    
    return top_movies[['movie_name', 'genre', 'rmse']]

# Function to plot the graph of movie relationships based on genre and RMSE
def plot_genre_based_graph(movie_genre_df, metrics_df, movie_name, top_n=5):
    # Get top recommended movies based on genre and RMSE
    top_genre_movies = get_top_movies_by_genre(movie_name, movie_genre_df, metrics_df, top_n)
    
    # Create a graph
    G = nx.Graph()

    # Add nodes with movie names and RMSE as attributes
    for _, row in top_genre_movies.iterrows():
        G.add_node(row['movie_name'], rmse=row['rmse'])
    
    # Add edges based on genre similarity (movies in the same genre)
    for i, row1 in top_genre_movies.iterrows():
        for j, row2 in top_genre_movies.iterrows():
            if i < j:
                genre1 = movie_genre_df[movie_genre_df['movie_name'] == row1['movie_name']]['genre'].values[0]
                genre2 = movie_genre_df[movie_genre_df['movie_name'] == row2['movie_name']]['genre'].values[0]
                
                if any(g in genre1.split('|') for g in genre2.split('|')):
                    G.add_edge(row1['movie_name'], row2['movie_name'])
    
    # Plot the graph
    plt.figure(figsize=(12, 10))
    
    # Normalize RMSE values to scale node sizes
    rmse_values = [G.nodes[node]['rmse'] for node in G.nodes]
    scaler = MinMaxScaler(feature_range=(500, 3000))  # Scale node sizes
    node_sizes = scaler.fit_transform([[val] for val in rmse_values]).flatten()
    
    # Normalize RMSE values for color gradient
    norm_rmse = [float(val) for val in rmse_values]  # Normalize the RMSE values
    max_rmse = max(norm_rmse)
    min_rmse = min(norm_rmse)
    rmse_range = max_rmse - min_rmse
    norm_rmse = [(rmse - min_rmse) / rmse_range for rmse in norm_rmse]  # Normalize RMSE to [0, 1]
    
    # Create a color map that transitions from green (low RMSE) to red (high RMSE)
    cmap = plt.cm.RdYlGn  # Red-Yellow-Green color map (inverted)
    node_colors = [cmap(rmse) for rmse in norm_rmse]
    
    # Define node positions using spring layout (force-directed layout)
    pos = nx.spring_layout(G, seed=42)  # Ensures a consistent layout each time
    
    # Draw the nodes and edges
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors)
    
    # Highlight the searched movie node
    searched_movie_node = movie_name
    if searched_movie_node in G:
        nx.draw_networkx_nodes(G, pos, nodelist=[searched_movie_node], node_size=3000, node_color='yellow')
    
    # Draw edges with color coding based on the searched movie
    edges = G.edges()
    edge_colors = ['lightgray' if searched_movie_node not in edge else 'red' for edge in edges]
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.5, edge_color=edge_colors)
    
    # Add movie name and RMSE as labels
    labels = {node: f"{node}\nRMSE: {G.nodes[node]['rmse']:.6f}" for node in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold')
    
    # Display the title
    plt.title(f'Movie Graph - Top {top_n} Movies in the Same Genre as \"{movie_name}\"', size=15)
    
    # Show the plot
    plt.axis('off')  # Turn off the axis
    plt.show()

# Main function to run the genre-based recommendation and visualization
if __name__ == '__main__':
    movie_genre_df = load_movie_data()  # Replace with your actual movie genre data file
    
    try:
        metrics_df = get_metrics(path='data/results/metrics')  # Replace with your actual metrics data path
        
        # Get top movie recommendations based on RMSE
        best_movie_recommendations = get_best_movie_recommendation_by_rmse(metrics_df, movie_genre_df, top_n=5)
        print(f"\nTop 5 movies based on RMSE:")
        print(tabulate(best_movie_recommendations, headers=["Movie", "RMSE"], tablefmt="grid"))
        
        # Now, let's say you want to get top movies based on genre for a specific movie
        movie_name = "Friday (1995)"  # Example movie for genre-based recommendations
        top_genre_movies = get_top_movies_by_genre(movie_name, movie_genre_df, metrics_df, top_n=5)
        print(f"\nTop 5 movies in the same genre as '{movie_name}':")
        print(tabulate(top_genre_movies, headers=["Movie", "Genre", "RMSE"], tablefmt="grid"))
        
        # Plot the graph of movie relationships based on genre and RMSE
        plot_genre_based_graph(movie_genre_df, metrics_df, movie_name="Friday (1995)", top_n=5)
    
    except KeyError as e:
        print(e)
    except ValueError as e:
        print(e)
