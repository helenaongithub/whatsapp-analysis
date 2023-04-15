# -*- coding: utf-8 -*-
import re
import pandas as pd
import matplotlib.pyplot as plt
import emoji

import nltk
from nltk.corpus import stopwords as nltk_stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download("vader_lexicon", quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

from textblob_de import TextBlobDE as TextBlob
from collections import Counter
from tabulate import tabulate
from wordcloud import WordCloud, STOPWORDS

from word_lists import system_messages


# Create a function to calculate sentiment scores for German text
def calculate_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Perform sentiment analysis on english messages
def sentiment_analysis(chat_language : str, df : pd.DataFrame):
    if chat_language == "eng":
        sia = SentimentIntensityAnalyzer()
        scores = df["Message"].apply(lambda x: sia.polarity_scores(x))
        df_scores = pd.DataFrame(list(scores))
        df = pd.concat([df, df_scores], axis=1)

        positive = df["pos"].sum()
        negative = df["neg"].sum()
        neutral = df["neu"].sum()
        
    # Perform sentiment analysis on german messages
    if chat_language == "ger":
        df["sentiment"] = df["Message"].apply(calculate_sentiment)
        positive = df[df["sentiment"] > 0].shape[0]
        negative = df[df["sentiment"] < 0].shape[0]
        neutral = df[df["sentiment"] == 0].shape[0]
        average = df["sentiment"].sum() / df.shape[0]

    if positive > negative:
        print("The overall sentiment is positive.")
        if chat_language == "eng": print(round(positive/df.shape[0], 4))
    elif positive < negative:
        print("The overall sentiment is negative.")
        if chat_language == "eng": print(round(negative/df.shape[0], 4))
    else:
        print("The overall sentiment is neutral.")
        
    if chat_language == "ger": print(f"The average value is {round(average, 4)}.\n")
    
   
# Extract all emojis in messages
def emojis_extraction(df : pd.DataFrame):
    emojis = []
    for message in df["Message"]:
        message = emoji.demojize(message)
        message = re.findall(r'(:[^:]*:)', message)
        if ": https:" in message:
            message.remove(": https:")
        list_emoji = [emoji.emojize(x) for x in message]
        emojis.extend(list_emoji)
    counter = Counter(emojis)
    
    # Get the most common elements and their counts
    most_common = counter.most_common()

    # Print the ranking
    table = []
    print("Ranking of most frequent emojis:")
    for i, (element, count) in enumerate(most_common):
        if i >= 10:
            break
        table.append([str(i+1)+".", element, count, "times"])
    print(tabulate(table, headers=[], tablefmt="plain"))
    
    


# Count the occurrences of each author
def count_messages(df : pd.DataFrame):
    print("\nRanking in terms of the number of messages written:")
    author_counts = {}
    for author in df["Author"]:
        if author not in author_counts:
            author_counts[author] = 1
        else:
            author_counts[author] += 1

    sorted_authors = sorted(author_counts.items(), key=lambda x: x[1], reverse=True)
    for author, count in sorted_authors:
        print(f"{author}: {count}")
    

# Create WordCloud
def create_wordcloud(message_str : str):
    irrelevant_words = ['omitted', 'audio', 'image', 'https', 'sticker', 'joined', 'deleted', 'using', 'invite', 'link', 'group', 's', 'vm']
    stopset = set(nltk_stopwords.words('german') + irrelevant_words)
    STOPWORDS.update(stopset)

    wordcloud = WordCloud(max_words=50, colormap="Pastel1").generate(message_str)

    width, height = wordcloud.to_image().size

    # create plot with the same size as the wordcloud
    fig = plt.figure(figsize=(width/70, height/70))

    # remove axes and border
    plt.axis("off")
    plt.tight_layout(pad=0)

    # display wordcloud
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.show()


if __name__ == "__main__":
    # Load chat log data
    chat_file = "chats/test_chat.txt"
    with open(chat_file, "r") as file:
        chat_log = file.readlines()

    # Extract chat messages
    chat_language = "ger"       #alternatively eng
    system_language = "eng"     #alternatively ger
    operation_system = "apple"  #alternatively android
    if system_language == "ger":
        if operation_system == "apple":
            pattern = r"\[(\d{2}\/\d{2}\/\d{4}) (\d{2}:\d{2}:\d{2}\]) ([^:]*): (.*)"
        if operation_system == "android":
            pattern = r"(\d{2}\.\d{2}\.\d{2}), (\d{2}:\d{2}) - ([^:]*): (.*)"
    elif system_language == "eng":
        pattern = r"\[(\d{2}\.\d{2}\.\d{2}), (\d{1,2}:\d{2}:\d{2} [AP]M)\] ([^:]*): (.*)"
    
    # Extract chat messages into a DataFrame
    messages = []
    for line in chat_log:
        irrelevant = False
        for system_message in system_messages:
            if system_message in line:
                irrelevant = True
        if irrelevant == True:
            continue        
        match = re.search(pattern, line)
        if match:
            date, time, author, message = match.groups()
            messages.append((date, time, author, message))
    df = pd.DataFrame(messages, columns=["Date", "Time", "Author", "Message"])
    
    # print("\n")
    # print(df["Date"])
    # print(df["Time"])
    # print("\n")

    
    # Extract all messages in one string
    message_str = ""
    for message in df["Message"]:
        message_str = message_str + " " + message        
    
    sentiment_analysis(chat_language, df)
    emojis_extraction(df)
    count_messages(df)
    create_wordcloud(message_str)    