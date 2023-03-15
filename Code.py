#Importing the libraries
import numpy as np
import pandas as pd

from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
 
df= pd.read_csv('./email spam.csv',encoding='ISO-8859-1') #creating a datframe
le=LabelEncoder()
data=df.to_numpy() #creturn a NumPy ndarray representing the values in given Series or Index.

#Slicing the array to get the message of mail and the prediction of Spam or not

mail=data[:,1]
prediction=data[:,0]

# importing the tokenizer, Stop Words and Stemmer to clean up the data
tokenizer=RegexpTokenizer('\w+')
sw=set(stopwords.words('english'))
ps=PorterStemmer()


#custom functions to clean up the data into smaller portions words and removing the insignificant (Stop Words)
def getStem(doc):
    doc=doc.lower()
    tokens=tokenizer.tokenize(doc) #breaking into small words
    removed_stopwords=[w for w in tokens if w not in sw]
    stemmed_words=[ps.stem(token) for token in removed_stopwords]
    clean_doc=' '.join(stemmed_words)
    return clean_doc
def getDoc(document):
    d=[]
    for doc in document:
        d.append(getStem(doc))
    return d
stemmed_doc=getDoc(mail)

# it is the method to convert the text into Numerical data which will be uses to create a VECTOR (BAGofWORDS) 
cv = CountVectorizer() 


Creating Vocabulary

#fit_transform to train the model
vc=cv.fit_transform(stemmed_doc) 
mail=vc.toarray()

 #train_test_split is used to estimate the performance of machine learning algorithms that are applicable for prediction-based
X_train, X_test, y_train, y_test = train_test_split(
...     mail, prediction, test_size=0.323, random_state=42)
train_test_split?


#Naive Bayes from Sklearn
from sklearn.naive_bayes import MultinomialNB 

#multinomialNB uses Naive Baye's and Baye's Theorem for probability,
model=MultinomialNB() 

#it calculates the likelihood of given sample happening 
model.fit(X_train,y_train)
model.score(X_test,y_test) #checks probabilty of accuracy


#Custom message check
def prepare(messages):
    d=getDoc(messages)
    #not fit_transform because creates new vocab only transform
    return cv.transform(d)
    
    
    
messages=["""

Dear Customer,
 
The last time I wrote to you was when India was in the middle of a difficult COVID wave. Telecom was essential to help you lead your life. Be it working from home, studying from home, being entertained or shopping online, our people were proud to serve you at that time of need.
 
Today, however, I am excited to be writing to you under happier circumstances. Within a few weeks, we will commence the launch of our next generation technology, Airtel 5G. Some of you have asked me questions about what 5G will do for you and how you can get it. Let me try and answer these questions for you.
 
1. What will Airtel 5G do for you?
 
Airtel 5G will deliver dramatically higher speeds compared to a 4G network. It could be anywhere between 20 to 30 times the speed you get today. This will allow you to boot up an application or download a heavy file in no time.
 
Airtel 5G will also enable differential quality for special requirements, something called network slicing. So, if you are a gamer, and want a flawless experience, we will be able to slice the network for you. Or if you are working from home and want a consistent experience, we will deliver it for you.
 
2. Why is Airtel 5G the best for you?
 
The Airtel 5G network is being built keeping in mind your smartphone and you. So there are three clear advantages.
 
First, of the two 5G technologies, we have chosen a specific 5G technology that has the widest eco-system in the world. This means that all 5G smartphones in India will work on the Airtel network without any glitch. This will be true even when you travel abroad with your Airtel 5G enabled phone. In other technologies, it is possible that as many as four out of ten 5G phones don't support 5G.
 
Second, we are confident of raising the bar on the experience we deliver to you. Over the last few years, our 4G network has been consistently rated the best in speed, video and gaming experience by independent rating agencies. We have used this strength to bring the expertise of our best engineers, built state-of-the-art tools and conducted numerous first-of-its-kind trials across several cities and use cases to ensure that your Airtel 5G experience is incomparable.
 
Finally, we will be kinder to the environment. All of us are now struggling with extreme heat and unpredictable rains caused by climate change. This problem is now real. So we have signed on to an ambitious goal of lowering our carbon footprint in the next few decades. As a result, the 5G solution we have chosen will be the most energy and carbon efficient in India.
 
3. When can you start experiencing 5G?
 
We expect to launch our 5G services within a month. By December, we should have coverage in the key metros. After that we will expand rapidly to cover the entire country. We expect to cover all of urban India by the end of 2023. If you want to know the availability of 5G in your town, you will be able to check it on the Airtel Thanks App and see whether your phone and city is 5G ready. This feature will be available on our app with the 5G launch.
 
4. Three easy steps for you to access Airtel 5G:
 
Most smartphones that are more than a year old do not have a 5G chipset. However, new smartphones that are now in India are mostly 5G enabled. So, if you are buying a new smartphone, do check whether it is 5G enabled.
 
Then enable 5G settings. In order to enable 5G on your phone, go to the settings tab and get to connections or mobile network. You will be shown a choice to pick 5G in addition to 4G or LTE. Select that mode and you are ready.
 
Your Airtel SIM is already 5G enabled. So it will work seamlessly on your 5G smart phone.
 
I look forward to any suggestions or feedback that you may have and thank you for giving us the privilege to serve you.
 
Sincerely,
 
Gopal Vittal
CEO Airtel""",
         
 """Hello,

It is sometimes said that the language we speak affects the way we think and see the world. That’s why it can be interesting to see what it looks like in other languages and to find the missing word that could change everything for us!

At this time of year, the word “rentrée” crops up a lot in France. This French noun literally means “return”, but, in this instance, refers to the start of September. The long summer vacation is over and it's time to go back to work, back to the office, back to school and normal life. "La rentrée" is a time for excited anticipation at the opportunities that lie ahead.

It is so significant that it has its own name – there is no equivalent word in English to express this feeling. But still, isn’t “rentrée” a strange word? Of course, it does not mean the same thing to everyone, but generally speaking, it implies to restart, to renew, to re-establish...

Our suggestion for this new month of September, then, is to break out of this cycle of getting back to it and to change our outlook: instead of a return, a restart, what if we were just getting ready for a simple start? The start of a new season, of a new energy, going on a path never taken before.

We will live each day of this new start only once, so let us enjoy the good days, and be relieved that the bad days are a one-off!

Meditating on the road
What do you usually do when you are on public transport? Do you read? Talk? Listen to music? Or… look frantically at your phone? We suggest you give meditation a go, just like at home in the quiet, but this time in the noise, in the middle of a crowd!"""    
         
        ,  
         
         """First, the surprise!

Here's a total of 40 free spins for you to use on our subsidiary casinos:
20 FREE spins on PlayBitcoinGames
20 FREE spins on PlayPerfectMoneyGames

NO new deposit required (only a $1 all-time, which qualifies you for similar codes we send out each month)

Stock up on extra BAP by buying Bulk Ads until the end of 31st August:
Get 20% extra BAP, 30% if buying with litecoin/bitcoin directly!

Don't forget about our amazing cashback offer on MyTrafficValue, by playing any of the games there (only lasts another 2 days):
- 100% cashback up to $5 + 20% extra up to $1000 (on any total game losses during event, applied at end of the period)
* $300 minimum wagering requirement to qualify

Greetings,

The PTCshare team
https://www.ptcshare.com"""
          ,
     """
Dear Candidate,

 Hurry up! Registration open for Company Secretary Executive Entrance Test (CSEET) November 2022 session. Last date of Registration is 15th October, 2022. For Registration click on https://smash.icsi.edu/Scripts/CSEET/Instructions_CSEET.aspx.
         """,
         """We noticed a new sign-in to your Google Account on a Mac device. If this was you, you don’t need to do anything. If not, we’ll help you secure your account.
"""]



messages=prepare(messages)

 #prediction of the model 
y_predict=model.predict(messages)
y_predict
