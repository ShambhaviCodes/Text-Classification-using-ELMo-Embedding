# Text-Classification-using-ELMo-Embedding

When we talk about supervised learning, a much exploited task is _'Text or Image Classification'_. Today we will discuss Text Classification on BBC News Dataset.

### Dataset
We’ll use a public dataset from the __BBC__ comprised of 2225 articles, each labeled under one of 5 categories: business, entertainment, politics, sport or tech.
The dataset is broken into 1490 records for training and 735 for testing. The goal will be to build a system that can accurately classify previously unseen news articles into the right category.

### Preprocessing
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
' in ' , ' out ' , ' on ' , ' off ' , ' over ' , ' under ' , ' again ' , ' further ' , ' then ' , ' once ' , ' here ' ,
' there ' , ' when ' , ' where ' , ' why ' , ' how ' , ' all ' , ' any ' , ' both ' , ' each ' , ' few ' , ' more ' ,
' most ' , ' other ' , ' some ' , ' such ' , ' no ' , ' nor ' , ' not ' , ' only ' , ' own ' , ' same ' , ' so ' ,
' than ' , ' too ' , ' very ' , ' s ' , ' t ' , ' can ' , ' will ' , ' just ' , ' don ' , ' should ' , ' now ' , ' d ' ,
' ll ' , ' m ' , ' o ' , ' re ' , ' ve ' , ' y ' , ' ain ' , ' aren ' , ' couldn ' , ' didn ' , ' doesn ' , ' hadn ' ,
' hasn ' , ' haven ' , ' isn ' , ' ma ' , ' mightn ' , ' mustn ' , ' needn ' , ' shan ' , ' shouldn ' , ' wasn ' ,
' weren ' , ' won ' , ' wouldn ' ]

We also encode our labels for the classification task.

### Embedding
_Word Embedding Model_ was a key breakthrough for learning representations for text where similar words have a similar representation in the vector space. 
_ELMo is a deep contextualized word representation that models both (1) complex characteristics of word use (e.g., syntax and semantics), and (2) how these uses vary across linguistic contexts (i.e., to model polysemy). These word vectors are learned functions of the internal states of a deep bidirectional language model (biLM), which is pre-trained on a large text corpus. They can be easily added to existing models and significantly improve the state of the art across a broad range of challenging NLP problems, including question answering, textual entailment and sentiment analysis._ - ELMo was developed by [Allen](https://allennlp.org/elmo).

Contextualized Word Representation is the representation of word which is heavily dependent on the surrounding words. ELMo takes into account all the text before generating an embedding to capture the semantics of the text.

__What is Model Polysemy?__
Polysemy is the capability of a word to possess more that one meaning.
_Bright_ means 'Shining' as well as 'Intelligent'.

ELMo address these problems of text data modeling.

I shall discuss more about different types of SOTA embeddings in another post.
ELMo Embedding pre-trained model trained on 1 Billion Word Benchmark is available on [Tensorflow-Hub](https://tfhub.dev/google/elmo/1).
Let's code!

```
import tensorflow as tf
import tensorflow_hub as hub
embed = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
def ELMoEmbedding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]
```
Training our model we achieve an accuracy of 0.91 and a categorical crossentropy loss of 0.28.
```
input_text = Input(shape=(1,), dtype=tf.string)
embedding = Lambda(ELMoEmbedding, output_shape=(1024, ))(input_text)
dense = Dense(256, activation='relu')(embedding)
pred = Dense(5, activation='softmax')(dense)
model = Model(inputs=[input_text], outputs=pred)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```
You can view the complete notebook [here](https://github.com/ShambhaviCodes/Text-Classification-using-ELMo-Embedding/blob/master/Text_Classification.ipynb).
