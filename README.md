# Text-Classification-using-ELMo-Embedding

When we talk about supervised learning, a much exploited task is _'Text or Image Classification'_. Today we will discuss Text Classification on BBC News Dataset.

#### Dataset
We’ll use a public dataset from the __BBC__ comprised of 2225 articles, each labeled under one of 5 categories: business, entertainment, politics, sport or tech.
The dataset is broken into 1490 records for training and 735 for testing. The goal will be to build a system that can accurately classify previously unseen news articles into the right category.

#### Preprocessing
We can not feed raw text or human understandable text directly to our model. Preprocessing a text involves multiple tasks such as stemming (breaking down) a word to its root, stopword removal to eliminate repetitive and redundant words or simply, stopwords. Preprocessing of text can not be generalised and is very specific to the task and domain of the data. Since our dataset is fairly simple and this is a beginner focussed tutorial we will remove stopwords for preprocessing our data.

We load our data using _Pandas_ :
```
import pandas as pd
data = pd.read_csv('Filename.csv')
```

We use __NLTK__ or Natural Language Toolkit, a Python library for modeling text. Stop words are the repetitive words, articles and conjunctions for example, which do not add value to the text from NLP perspective. NLTK library makes our task easy by providing us a list of commonly occuring stopwords. 

```
from nltk.corpus import stopwords
stop_words = stopwords.words( ' english ' )
print(stop_words)
```

Output : <br />
[ ' i ' , ' me ' , ' my ' , ' myself ' , ' we ' , ' our ' , ' ours ' , ' ourselves ' , ' you ' , ' your ' , ' yours ' ,
' yourself ' , ' yourselves ' , ' he ' , ' him ' , ' his ' , ' himself ' , ' she ' , ' her ' , ' hers ' ,
' herself ' , ' it ' , ' its ' , ' itself ' , ' they ' , ' them ' , ' their ' , ' theirs ' , ' themselves ' ,
' what ' , ' which ' , ' who ' , ' whom ' , ' this ' , ' that ' , ' these ' , ' those ' , ' am ' , ' is ' , ' are ' ,
' was ' , ' were ' , ' be ' , ' been ' , ' being ' , ' have ' , ' has ' , ' had ' , ' having ' , ' do ' , ' does ' ,
' did ' , ' doing ' , ' a ' , ' an ' , ' the ' , ' and ' , ' but ' , ' if ' , ' or ' , ' because ' , ' as ' , ' until ' ,
' while ' , ' of ' , ' at ' , ' by ' , ' for ' , ' with ' , ' about ' , ' against ' , ' between ' , ' into ' ,
' through ' , ' during ' , ' before ' , ' after ' , ' above ' , ' below ' , ' to ' , ' from ' , ' up ' , ' down ' ,
' in ' , ' out ' , ' on ' , ' off ' , ' over ' , ' under ' , ' again ' , ' further ' , ' then ' , ' once ' , ' here ' ,5.5. Tokenization and Cleaning with NLTK
44
' there ' , ' when ' , ' where ' , ' why ' , ' how ' , ' all ' , ' any ' , ' both ' , ' each ' , ' few ' , ' more ' ,
' most ' , ' other ' , ' some ' , ' such ' , ' no ' , ' nor ' , ' not ' , ' only ' , ' own ' , ' same ' , ' so ' ,
' than ' , ' too ' , ' very ' , ' s ' , ' t ' , ' can ' , ' will ' , ' just ' , ' don ' , ' should ' , ' now ' , ' d ' ,
' ll ' , ' m ' , ' o ' , ' re ' , ' ve ' , ' y ' , ' ain ' , ' aren ' , ' couldn ' , ' didn ' , ' doesn ' , ' hadn ' ,
' hasn ' , ' haven ' , ' isn ' , ' ma ' , ' mightn ' , ' mustn ' , ' needn ' , ' shan ' , ' shouldn ' , ' wasn ' ,
' weren ' , ' won ' , ' wouldn ' ]
