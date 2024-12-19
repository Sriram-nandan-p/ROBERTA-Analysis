# Stress and Anxiety Sentiment Analysis

This project analyzes customer reviews to determine the sentiment (Positive, Neutral, or Negative) related to **stress** and **anxiety relief** using a fine-tuned RoBERTa model. It processes a dataset of reviews, filters relevant ones based on specific keywords, performs sentiment analysis, and visualizes the results.

---

## Features
1. **Sentiment Analysis**:
   - Uses a fine-tuned RoBERTa model (`cardiffnlp/twitter-roberta-base-sentiment`) to classify reviews into Positive, Neutral, or Negative sentiments.
   - Focuses on reviews containing keywords such as "stress," "anxiety," "relax," "calm," or "relieve."

2. **Output**:
   - Generates an output CSV file (`Stress_Anxiety_Sentiment_Analysis.csv`) with the original reviews and their sentiment labels.

3. **Visualization**:
   - A separate script calculates sentiment percentages and generates a bar chart (`Sentiment_Distribution_Graph.png`) to visualize the sentiment distribution.

---

## File Overview
- `sentiment_analysis.py`: Main script for filtering reviews, performing sentiment analysis, and saving the results.
- `sentiment_percentage_graph.py`: Script to calculate sentiment percentages and generate a bar chart based on the analysis output.
- `Merged_Walmart_Reviews.csv`: Input dataset containing customer reviews (replace this with your own dataset).

---

## Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Required Python libraries:
  ```bash
  pip install torch transformers datasets pandas matplotlib
