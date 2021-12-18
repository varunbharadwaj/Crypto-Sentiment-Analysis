# Crypto-Sentiment-Analysis
Sentiment analysis on telegram messages related to cryptocurrency. <br>
We analyze telegram messages on cryptocurrency in the period from May 1 2021 to May 15 2021. This period had a sharp increase in the shiba/doge coin prices and we compare the sentiments of telegram messages to find patterns in this period.

## Results

### Number of messages per day
![message_count_plot](https://user-images.githubusercontent.com/17334869/146632562-68de5ed5-91b1-4406-a0a8-b8bb7c018c69.png)

### Average sentiment per day
![sentiment_plot](https://user-images.githubusercontent.com/17334869/146632568-0a43d54a-13fb-497d-b792-e7516098d65a.png)

We can see a peak in the number of messages around May 7 - May 11 and at the same time a dip in the average sentiment in the same period. This corresponds to a sharp increase in dogecoin and shiba inu prices in this duration which led to lot more discussion on telegram. People discuss about "Elon Musk tweeting and pumping it" corresponding to an increase in price until May 7 after which it dropped. They also discuss about their disappointment with accessibility issues, no response from the support team, scams and not being able to sell their coins. We can also see this trend in the sentiment plot with a dip around this time period. 

## Approach

1. The telegram message dump can be found under the 'data' directory.
2. We pre-process the messages by removing non-ascii/non-alphabetic characters and by converting all messages to lowercase. We then remove extra spaces in the messages.
3. We then filter messages mentioning either DOGE or SHIBA. We also only retain messages in the English language.
4. We compute the sentiment scores of messages per day and generate plots for sentiment and message counts from May 1 to May 15 2021.

## Libraries

1. NLTK: used for Sentiment analysis
2. Spacy: used for language detection
3. Plotly: for plotting graphs 

## Steps to reproduce the results

1. Install dependencies
```
  pip install -r requirements.txt
```

2. Run the following script to generate results. Python 3.9.9 was used in this project.
```
python3 crypto_sentiment_analysis.py
```
