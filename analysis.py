# -*- coding: utf-8 -*-
import re
import pandas as pd
import matplotlib.pyplot as plt
import os
import emoji
import datetime as dt

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

from word_lists import system_messages, irrelevant_words


# Create a function to calculate sentiment scores for German text
def calculate_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Removes emoji of a string
def remove_emoji(string):
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U00010000-\U0010ffff"
            u"\u2600-\u2B55"
            u"\u200d"
            u"\u23cf"
            u"\u23e9"
            u"\u231a"
            u"\ufe0f"                 # dingbats
            u"\u3030"
                               "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', string)

# Changes the order of the dates
def change_date_format(df : pd.DataFrame):
    df["Date"] = df["Date"].apply(lambda x: x.split(",")[0])#.replace("/", ".")
    for i in range(len(df)):
        # Split the date string into year, month, and day
        day, month, year = df.loc[i, "Date"].replace("/",".").split(".")
        
        # Concatenate the year, month, and day in the desired order
        new_date = "{}-{}-{}".format(year, month, day)
        
        # Update the DataFrame with the new date format
        df.loc[i, "Date"] = new_date
    
    return df

def change_time_format(df : pd.DataFrame, system_language : str, operation_system : str):
    for i in range(len(df)):
        if system_language == "ger":
            if operation_system == "IOS":
                time_obj = dt.datetime.strptime(df.loc[i, "Time"], '%H:%M:%S').time()
                time_str = time_obj.strftime('%H:%M')
                df.loc[i, "Time_obj"] = time_obj
                df.loc[i, "Time"] = time_str
            elif operation_system == "android":
                time_obj = dt.datetime.strptime(df.loc[i, "Time"], '%H:%M').time()
                df.loc[i, "Time_obj"] = time_obj
        elif system_language == "eng":
                time_obj = dt.datetime.strptime(df.loc[i, "Time"], '%I:%M:%S %p').time() 
                time_str = time_obj.strftime('%H:%M')
                df.loc[i, "Time_obj"] = time_obj
                df.loc[i, "Time"] = time_str
    return df                
            # time_obj = datetime.strptime(time_string, '%H:%M').time()
            # format_time = datetime.time(3, 12, 24, 10)
        
    # 20:22:06 sys ger oper IOS
    # 12:27 sys ger oper android
    # 8:17:43 AM sys eng 
    
def get_number_of_messages_per_author(df : pd.DataFrame):
    author_counts = {}
    for author in df["Author"]:
        if author not in author_counts:
            author_counts[author] = 1
        else:
            author_counts[author] += 1
    return author_counts 


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
def count_messages_descending(df : pd.DataFrame):
    print("\nRanking in terms of the number of messages written:")
    author_counts = get_number_of_messages_per_author(df)
    sorted_authors = sorted(author_counts.items(), key=lambda x: x[1], reverse=True)
    for author, count in sorted_authors:
        print(f"{author}: {count}")


# Create WordCloud
def create_wordcloud(message_str : str):
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
    path = os.path.join("plots", "wordcloud")
    plt.savefig(path)
    
# Create User WordCloud
def create_user_wordcloud(df,author:str):
    usertext=" ".join(list(df.loc[df["Author"]==author]["Message"]))
    create_wordcloud(usertext)

# Creates a timeline depicting the number of messages chat participants write in a day
def timeline(df : pd.DataFrame):
    df["Author"] = df["Author"].apply(remove_emoji)
    
    # count the number of messages per author per day
    df_timeline = df.groupby(["Date", "Author"]).count()["Message"]
    df_timeline = pd.DataFrame(df_timeline).reset_index()
    
    # get the 6 authors with the highest message counts
    top_authors = df_timeline.groupby("Author").sum()["Message"].nlargest(9).index
    df_timeline = df_timeline[df_timeline["Author"].isin(top_authors)]
    df_timeline = df_timeline.sort_values("Date")
    
    # pivot the table to create a table with the dates as the index and each author's message count as a column
    df_timeline = df_timeline.pivot(index="Date", columns="Author", values="Message")
    
    ax = df_timeline.plot(kind="line", marker="o", figsize=(14, 8), markersize=5)
    ax.set_title("Timeline of Chat Activity")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Messages")    
    path = os.path.join("plots", "timeline")
    plt.savefig(path)
    
def activity_per_hour(df : pd.DataFrame):
    df["Author"] = df["Author"].apply(remove_emoji)
    df["Hour"] = df["Time_obj"].apply(lambda x: x.hour)
    
    # Count the number of messages for each hour
    hour_counts = df.groupby("Hour")["Message"].count()
    
    # Create a bar chart of the message counts
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(hour_counts.index, hour_counts.values)
    ax.set_title("Chat Activity by Hour")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Number of Messages")
    ax.set_xticks(range(24))
    path = os.path.join("plots", "activity_per_hour")
    plt.savefig(path)

def activity_per_hour_and_author(df : pd.DataFrame):
    df["Author"] = df["Author"].apply(remove_emoji)
    df["Hour"] = df["Time_obj"].apply(lambda x: x.hour)
    author_counts = df.groupby(["Hour", "Author"])["Message"].count().reset_index()
    pivot_table = author_counts.pivot(index="Hour", columns="Author", values="Message")
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot_table.plot(kind="bar", ax=ax, stacked=True)
    ax.set_title("Chat Activity by Hour and Author")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Number of Messages")
    ax.set_xticks(range(24))
    path = os.path.join("plots", "activity_per_hour_and_author")
    plt.savefig(path)
    


if __name__ == "__main__":
    os.makedirs("chats", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    # Load chat log data
    chat_file = "chats/test_chat.txt"
    with open(chat_file, "r") as file:
        chat_log = file.readlines()

    # Extract chat messages
    chat_language = "ger"       #alternatively eng
    system_language = "eng"     #alternatively ger
    operation_system = "IOS"  #alternatively android
    if system_language == "ger":
        if operation_system == "IOS":
            pattern = r"\[(\d{2}\/\d{2}\/\d{4}) (\d{2}:\d{2}:\d{2})\] ([^:]*): (.*)"
            # 20:22:06 sys ger oper IOS
        if operation_system == "android":
            pattern = r"(\d{2}\.\d{2}\.\d{2}), (\d{2}:\d{2}) - ([^:]*): (.*)"
            # 12:27 sys ger oper android
    elif system_language == "eng":
        pattern = r"\[(\d{2}\.\d{2}\.\d{2}), (\d{1,2}:\d{2}:\d{2} [AP]M)\] ([^:]*): (.*)"
            # 8:17:43 AM sys eng 
    
    
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
    
    # Extract all messages in one string
    message_str = ""
    for message in df["Message"]:
        message_str = message_str + " " + message        
    
    df = change_date_format(df)
    df = change_time_format(df, system_language, operation_system)
    
    plt.style.use('dark_background')
    
    sentiment_analysis(chat_language, df)
    emojis_extraction(df)
    count_messages_descending(df)
    create_wordcloud(message_str)  
    timeline(df)
    if len(get_number_of_messages_per_author(df)) > 5:
        activity_per_hour(df)
    else:
        activity_per_hour_and_author(df)  