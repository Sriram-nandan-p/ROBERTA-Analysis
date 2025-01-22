import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

# Load dataset
cleaned_data_path = "Stress_Anxiety_Sentiment_Analysis.csv"
df = pd.read_csv(cleaned_data_path)

# Map sentiment labels to numerical values
sentiment_mapping = {'Positive': 2, 'Neutral': 1, 'Negative': 0}
df['Sentiment_Label'] = df['Sentiment'].map(sentiment_mapping)

# Simulated model predictions (replace this with actual predictions if available)
np.random.seed(42)
df['Predicted_Sentiment'] = df['Sentiment_Label']

# Introduce slight misclassification for analysis
for i in range(len(df)):
    if np.random.rand() < 0.05:  # Simulate 5% misclassification
        df.loc[i, 'Predicted_Sentiment'] = np.random.choice([0, 1, 2])

# Generate Confusion Matrix
conf_matrix = confusion_matrix(df['Sentiment_Label'], df['Predicted_Sentiment'])

# Plot Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Neutral', 'Positive'], 
            yticklabels=['Negative', 'Neutral', 'Positive'])
plt.xlabel('Predicted Sentiment')
plt.ylabel('Actual Sentiment')
plt.title('Confusion Matrix for Sentiment Classification')
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()
