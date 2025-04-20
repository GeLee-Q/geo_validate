import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def compare_scores_between_files(csv_file1, csv_file2, display_points=32, output_image="score_comparison.png"):
    # Read the two CSV files
    # Assuming both files have headers with a column named 'score'
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)
    
    # Assuming the first column is the index column
    # Extract the index column name from the first file
    index_col = df1.columns[0]
    
    # Rename the score columns to distinguish them
    df1 = df1.rename(columns={'score': 'score_file1'})
    df2 = df2.rename(columns={'score': 'score_file2'})
    
    # Merge the two dataframes using the index column
    merged_df = pd.merge(df1, df2, on=index_col, how='outer')
    
    # Calculate the score difference (score_file2 - score_file1)
    merged_df['difference'] = merged_df['score_file2'] - merged_df['score_file1']
    
    # Calculate means using ALL data points
    mean_score1 = merged_df['score_file1'].mean()
    mean_score2 = merged_df['score_file2'].mean()
    mean_difference = mean_score2 - mean_score1
    
    # Create a subset for visualization (first display_points)
    display_df = merged_df.head(display_points)
    
    # Create the visualization
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot the comparison of scores in the first subplot (first display_points only)
    sns.lineplot(x=index_col, y='score_file1', data=display_df, marker='o', label=f'File 1 Score', ax=ax1)
    sns.lineplot(x=index_col, y='score_file2', data=display_df, marker='x', label=f'File 2 Score', ax=ax1)
    ax1.set_title(f'Score Comparison (First {display_points} Points)')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Score')
    ax1.legend()
    ax1.grid(True)
    
    # Plot the score differences in the second subplot (first display_points only)
    bars = sns.barplot(x=index_col, y='difference', data=display_df, ax=ax2)
    ax2.set_title(f'Score Difference (First {display_points} Points)')
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Difference')
    ax2.grid(True)
    plt.setp(ax2.get_xticklabels(), rotation=45)
    
    # Add a horizontal line at zero to highlight positive/negative differences
    ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # Add a third subplot to display means (calculated from ALL data) and their difference
    mean_data = pd.DataFrame({
        'Dataset': ['File 1 (All Data)', 'File 2 (All Data)', 'Difference'],
        'Value': [mean_score1, mean_score2, mean_difference]
    })
    bars = sns.barplot(x='Dataset', y='Value', data=mean_data, ax=ax3)
    ax3.set_title('Mean Scores (All Data) and Difference')
    ax3.set_ylabel('Value')
    # Add text labels on top of bars
    for i, bar in enumerate(bars.patches):
        bars.text(
            bar.get_x() + bar.get_width()/2.,
            bar.get_height() + 0.01,
            f'{mean_data["Value"].iloc[i]:.4f}',
            ha='center'
        )
        
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_image, dpi=300)
    print(f"Visualization saved as '{output_image}'")
    
    # Display the figure
    plt.show()
    
    # Print detailed statistics
    print("\nMean Comparison (All Data Points):")
    print(f"Total number of data points: {len(merged_df)}")
    print(f"Mean score in File 1: {mean_score1:.4f}")
    print(f"Mean score in File 2: {mean_score2:.4f}")
    print(f"Mean difference (File 2 - File 1): {mean_difference:.4f}")
    print(f"Percentage difference: {mean_difference/mean_score1*100:.2f}% of File 1 mean")
    
    return merged_df, mean_score1, mean_score2, mean_difference

# Example usage
if __name__ == "__main__":
    # Replace with your CSV file paths
    root_path = r"/sgl-workspace/validation/geo_validate/result/engine/"
    csv_file1 = root_path + r"sdpa/evaluation_results_engine.csv"
    csv_file2 = root_path + r"flash_attn/evaluation_results_engine.csv"
    
    # Optional: Specify a custom output image filename
    output_image = root_path + r"score_comparison_results.png"
    
    # Compare all data but display only first 32 points in visualization
    merged_df, mean1, mean2, mean_diff = compare_scores_between_files(
        csv_file1, csv_file2, display_points=32, output_image=output_image
    )