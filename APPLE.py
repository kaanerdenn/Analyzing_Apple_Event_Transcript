import csv
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer  # Stemming
from collections import Counter
from wordcloud import WordCloud
from youtube_transcript_api import YouTubeTranscriptApi
from textblob import TextBlob  # Sentiment Analysis

# YouTube video URL
video_url = "https://www.youtube.com/live/ZiP1l7jlIIA?si=u02bXcg6x_D7xV_0"


# Transcribe the YouTube video
def transcribe_youtube_video(url):
    try:
        srt = YouTubeTranscriptApi.get_transcript(url)
        return srt
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


def save_transcript_to_csv(transcript, output_file):
    if transcript:
        with open(output_file, 'w', newline='') as csvfile:
            fieldnames = ['text', 'start', 'duration']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for entry in transcript:
                writer.writerow({'text': entry['text'], 'start': entry['start'], 'duration': entry['duration']})
        print(f"Transcript saved to {output_file}")
    else:
        print("No transcript available to save.")


def preprocess_text(text):
    # Tokenize and preprocess the text
    nltk.download('punkt')
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    # Tokenize the text
    words = word_tokenize(text.lower())

    # Remove special characters, numbers, and stopwords
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]

    # Apply stemming
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_words]

    return stemmed_words


def analyze_transcript(transcript_file):
    # Load transcript data into a pandas DataFrame
    data = pd.read_csv(transcript_file)

    # Combine and preprocess the text
    text = ' '.join(data['text'])
    preprocessed_words = preprocess_text(text)

    # Calculate word frequencies
    word_freq = Counter(preprocessed_words)

    # Generate and display a word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(preprocessed_words))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud')
    plt.show()

    # Create a bar chart of word frequencies
    most_common = word_freq.most_common(10)
    plt.figure(figsize=(12, 6))
    plt.bar(*zip(*most_common))
    plt.xticks(rotation=45)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Top 10 Most Common Words')
    plt.show()

    # Histogram of word lengths
    word_lengths = [len(word) for word in preprocessed_words]
    plt.figure(figsize=(8, 4))
    plt.hist(word_lengths, bins=20, edgecolor='k')
    plt.xlabel('Word Length')
    plt.ylabel('Frequency')
    plt.title('Distribution of Word Lengths')
    plt.show()

    # Sentiment Analysis
    sentences = sent_tokenize(text)
    sentiment_scores = [TextBlob(sentence).sentiment.polarity for sentence in sentences]

    # Create a line chart for sentiment analysis
    plt.figure(figsize=(10, 5))
    plt.plot(sentiment_scores)
    plt.xlabel('Sentence')
    plt.ylabel('Sentiment Polarity')
    plt.title('Sentiment Analysis')
    plt.show()


if __name__ == "__main__":
    output_csv = "transcript.csv"
    transcribe_youtube_video(video_url)
    analyze_transcript(output_csv)
