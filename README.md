# IPL Match Analysis & Winner Prediction

A data science project analyzing 756 IPL matches (2008-2019) to find patterns 
and predict match winners using Machine Learning.

## Tools Used
- Python (Google Colab)
- Pandas (data analysis)
- Matplotlib (visualizations)
- Scikit-learn (machine learning)

## Key Findings
- Mumbai Indians have the most wins (109), but Chennai Super Kings have a better win rate (61% vs 58%)
- Winning the toss has minimal impact on match outcome (only 52% of toss winners go on to win)
- Eden Gardens is the most used IPL venue (77 matches hosted)
- CH Gayle leads Player of the Match awards with 21 wins
- 4 matches were abandoned with no result (removed before ML modeling)

  ## Machine Learning Model
- Algorithm: Decision Tree Classifier
- Features used: Team 1, Team 2, Venue, Toss Winner, Toss Decision
- Train/Test Split: 80/20
- Accuracy: 48%

The model's limited accuracy is expected — predicting cricket match outcomes 
from pre-match info alone is genuinely difficult. Features like team form, 
player availability, and pitch conditions would improve accuracy significantly.

## Project Structure
- `day1_loading_data.ipynb` - Loading and exploring the dataset
- `day2_value_counts.ipynb` - Analyzing wins and venues
- `day3_toss_vs_winner.ipynb` - Toss impact analysis
- `day4_filtering_data.ipynb` - Data filtering techniques
- `day5_groupby.ipynb` - Groupby and aggregations
- `day6_clean_average.ipynb` - Cleaning misleading averages
- `day7_win_percentage.ipynb` - Quick win percentage look
- `day8_win_percentage.ipynb` - True win percentage calculation
- `day9_top_players.ipynb` - Player of the Match analysis
- `day10_season_trends.ipynb` - Season-wise match counts
- `day11_first_bar_chart.ipynb` - First visualization
- `day12_pie_chart_and_labels.ipynb` - Pie charts and labels
- `day13_features_and_target.ipynb` - Defining ML features
- `day14_encoding_features.ipynb` - Label encoding
- `day15_first_model_training.ipynb` - Training the model
- `day16_model_predictions.ipynb` - Predictions and accuracy
