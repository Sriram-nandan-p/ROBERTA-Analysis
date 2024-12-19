
import pandas as pd
import matplotlib.pyplot as plt

# Load the output CSV file
file_path = "./Stress_Anxiety_Sentiment_Analysis.csv"
data = pd.read_csv(file_path)

# Check if the 'Sentiment' column exists
if "Sentiment" not in data.columns:
    raise ValueError("The output file must contain a 'Sentiment' column.")

# Calculate percentages
sentiment_counts = data["Sentiment"].value_counts(normalize=True) * 100
print("Sentiment Percentages:")
print(sentiment_counts)

# Plot the sentiment distribution
plt.figure(figsize=(8, 6))
sentiment_counts.plot(kind="bar", color=["red", "blue", "green"], alpha=0.7)
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Percentage")
plt.xticks(rotation=0)
plt.tight_layout()

# Save the graph
output_graph = "./Sentiment_Distribution_Graph.png"
plt.savefig(output_graph)
print(f"Graph saved as {output_graph}")

# Show the graph
plt.show()
