# Whatsapp-Analysis
This is a Python script to analyze WhatsApp chats exported in txt format. The script supports both English (at the moment only IOS) and German languages.

The following functionalities are provided:

- Calculate the overall sentiment of the chat
- Count emojis occurences
- Count the number of messages per author
- Count first Chat per Author
- Generate a word cloud
- Display a message timeline
- Create a message frequency histogram

## Dependencies
The following packages are required to run this script:

- pandas
- matplotlib
- emoji
- nltk
- textblob-de
- tabulate
- wordcloud

## Usage of the Tool

To use the script, follow these steps:

1. Export the WhatsApp chat you want to analyze as a txt file.
2. Copy the txt file to the same directory as the script, in a folder called chats.
3. Run the script in a Python environment.

## Functionalities
### Calculate Overall Sentiment
The script calculates the overall sentiment of the chat using sentiment analysis. The sentiment analysis algorithm used is the VADER algorithm for English texts and a custom algorithm for German texts. The overall sentiment is displayed as positive, negative, or neutral.


### Count Emojis Occurrences
The script counts the number of times emojis were used in the chat. The result is displayed in a table.
First the total number and then connected to the responsible authors.


### Count Messages per Author
The script counts the number of messages each author has sent in the chat. The result is displayed in a table.


### Count first Chat per Author
Count the number of times each author was the first to send a message on a given day


### Generate Word Cloud
The script generates a word cloud based on the messages in the chat. Stopwords and irrelevant words such as system messages are filtered out.


### Display Message Timeline
The script creates a timeline of the messages in the chat. The timeline is displayed as a scatter plot.


### Create Message Frequency Histogram
The script creates a histogram of the message frequency in the chat.


## Customization
The script can be customized to support other languages by changing the sentiment analysis algorithm and stopwords list. The system messages and irrelevant words lists can also be customized for specific chat types.