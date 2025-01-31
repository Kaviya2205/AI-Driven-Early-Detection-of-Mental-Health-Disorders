# AI-Driven Early Detection of Mental Health Disorders

## Project Overview
This project uses Natural Language Processing (NLP) and Machine Learning (ML) to detect early signs of mental health disorders from text inputs. It applies text preprocessing, TF-IDF vectorization, and a Naive Bayes classifier to classify text as indicative of mental health conditions.

## Features
- Text preprocessing (cleaning, stopword removal, tokenization)
- TF-IDF vectorization for feature extraction
- Naive Bayes classification for detection
- Model evaluation using accuracy and classification reports
- Trained model and vectorizer saved for future use

## Dataset
- The dataset should contain two columns:
  - `text`: User input text (e.g., social media posts, responses to surveys)
  - `label`: Classification label (e.g., 0 for normal, 1 for signs of mental health issues)
- Example dataset structure:
  
  | text | label |
  |------|-------|
  | "I feel very anxious today." | 1 |
  | "I had a great day at work!" | 0 |

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/mental-health-ai.git
   cd mental-health-ai
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
1. Run the script to train and evaluate the model:
   ```sh
   python mental_health_ai.py
   ```
2. The trained model (`mental_health_model.pkl`) and TF-IDF vectorizer (`tfidf_vectorizer.pkl`) will be saved.

## Future Enhancements
- Integration with a chatbot for real-time analysis
- Expanding the dataset for better accuracy
- Implementing deep learning models like LSTMs for better text classification

## Contributing
Feel free to submit pull requests or report issues.

## License
MIT License

