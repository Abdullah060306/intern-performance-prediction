# Intern Feedback Sentiment Analysis

## Objective
Analyze intern feedback to understand positive and negative sentiments using Machine Learning.

## Models Used
- **Logistic Regression** (TF-IDF + LR)
- **Transformers (BERT)** - Pre-trained model for comparison

## Dataset
- 65 intern feedback entries
- 25 Positive, 25 Negative, 15 Neutral
- Balanced dataset for fair training

## Key Results
- Logistic Regression Accuracy: ~98% on training
- Feature importance shows top positive words: "helped", "amazing", "great", "projects"
- Model successfully classifies: 🔴 Negative / ⚪ Neutral / ⭐ Positive

## Insights
1. **Mentorship** is the #1 driver of positive sentiment
2. **Workload** causes most negative feedback
3. **Regular feedback** keeps interns satisfied
4. **Recognition** significantly boosts morale

## How to Run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python main.py