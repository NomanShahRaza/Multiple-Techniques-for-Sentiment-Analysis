"""
@author: Noman Raza Shah
"""
#%% ====================
#   |Sentiment Analysis|
#   ====================
# Sentiment Analysis
# Sentiment analysis is basically the process of determining the attitude or the emotion of the writer, 
# i.e., whether it is positive or negative or neutral. The sentiment function of textblob returns two 
# properties, polarity, and subjectivity.

# Polarity is float which lies in the range of [-1,1] where 1 means positive statement and -1 means a negative statement. 
# Subjective sentences generally refer to personal opinion, emotion or judgment whereas objective refers to factual information. 
# Subjectivity is also a float which lies in the range of [0,1].


# https://github.com/jess-data/Twitter-2020-Sentiment-Analysis/blob/master/Twitter%20Sentiment%20Analysis%20Project.ipynb
# https://neptune.ai/blog/sentiment-analysis-python-textblob-vs-vader-vs-flair


#%%  Transformer Model
def sentiment_transformer(text):
    from transformers import pipeline
    classifier = pipeline('sentiment-analysis')
    return classifier(text)

#Textblob
def sentiment_textblob(text):
    from textblob import TextBlob
    def getSubjectivity(text):
        return TextBlob(text).sentiment.subjectivity
    def getPolarity(text):
        return TextBlob(text).sentiment.polarity
    def getSentimentTextBlob(polarity):
        if polarity < 0:
            return "Negative"
        elif polarity == 0:
            return "Neutral"
        else:
            return "Positive"
    Subjectivity=getSubjectivity(text)
    Polarity=getPolarity(text)
    SentimentTextBlob=getSentimentTextBlob(Polarity)
    return SentimentTextBlob

# NLTK (VADER) 
# https://towardsdatascience.com/the-best-python-sentiment-analysis-package-1-huge-common-mistake-d6da9ad6cdeb
def sentiment_NLTK(text):
    from nltk.sentiment import SentimentIntensityAnalyzer
    import numpy as np
    import operator
    import nltk
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(text)["compound"]
    def getSentimentTextBlob(polarity):
        if polarity < 0:
            return "Negative"
        elif polarity == 0:
            return "Neutral"
        else:
            return "Positive"    
    sentiment = getSentimentTextBlob(score)
    return sentiment

# Flair
# https://towardsdatascience.com/the-best-python-sentiment-analysis-package-1-huge-common-mistake-d6da9ad6cdeb
# !pip install --user flair
def sentiment_Flair(text):  
    '''Flair is a pre-trained embedding-based model. This means that each word is represented inside a vector space. 
    Words with vector representations most similar to another word are often used in the same context. 
    This allows us, to, therefore, determine the sentiment of any given vector, and therefore, any given sentence.''' 
    from flair.models import TextClassifier
    from flair.data import Sentence
    sia = TextClassifier.load('en-sentiment')
    def flair_prediction(x):
        sentence = Sentence(x)
        sia.predict(sentence)
        score = sentence.labels[0]
        print(score)
        if "POSITIVE" in str(score):
            return "POSITIVE"
        elif "NEGATIVE" in str(score):
            return "NEGATIVE"
        else:
            return "NEUTRAL"
    sentiment = flair_prediction(text)
    return sentiment

#%%
# text = "It is a bad habit" # negative
# text = "It is a not bad habit" # positive
# text = "It pretty hot in the summer" # positive
# text = "I’m not sure if I like the new design" 
# text = "I really like the new design of your website" # positive
# text = "@AmericanAir just landed - 3hours Late Flight - and now we need to wait TWENTY MORE MINUTES for a gate! I have patience but none for incompetence." # negative
text = "@AmericanAir we have 8 ppl so we need 2 know how many seats are on the next flight. Plz put us on standby for 4 people on the next flight?" # neutral

#%%
print("="*70)
print("TextBlob")
print(text)
result_textblob=sentiment_textblob(text)
print('This sentence is', result_textblob)
print("="*70)

#%%
print("="*70)
print("VANDER")
print(text)
result_NLTK=sentiment_NLTK(text)
print('This sentence is', result_NLTK)
print("="*70)

#%%
print("="*70)
print("Flair")
print(text)
result_Flair=sentiment_Flair(text)
print('This sentence is', result_Flair)
print("="*70)

#%% 
print("="*70)
print("Transformer")
print(text)
result_transformer=sentiment_transformer(text)
print('This sentence is', result_transformer[0].get("label"))
print("="*70)










