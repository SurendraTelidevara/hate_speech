import tensorflow_hub as hub
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

import numpy as np
import pandas as pd

data_file = 'train.csv'
df = pd.read_csv(data_file)

df.text = df.text.str.replace('https?://[A-Za-z0-9./]+', '')
df.text = df.text.str.replace('@[a-zA-Z0-9_]+', '')
df.text = df.text.str.replace('#[a-zA-Z0-9_]+', '')
df.text = df.text.str.replace('&amp;', 'and')
df.text = df.text.str.lower()
df.text = df.text.str.replace("[']", '')
df.text = df.text.str.replace('[^a-zA-Z]', ' ')

clean_text = []
for tweet in df['text']:
  clean_text.append(' '.join(tweet.split()))

embeddings = embed(clean_text)

embeddings.numpy()

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

X_train, X_test, y_train, y_test = train_test_split(embeddings.numpy(), df['labels'].values, test_size=0.33, random_state=42)

from sklearn.svm import SVC
clf = SVC(C=28, gamma='auto')
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print(f1_score(y_test, predictions))
print(accuracy_score(y_test, predictions))



"""# Formal (Embeddings)"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# # !wget https://aicrowd-production.s3.eu-central-1.amazonaws.com/dataset_files/challenge_287/1fe720be-90e4-4e06-9b52-9de93e0ea937_train.csv
# # !wget https://aicrowd-production.s3.eu-central-1.amazonaws.com/dataset_files/challenge_287/f6eb0bd7-6063-4e50-baa0-111feda638fb_test.csv
# 
# !wget https://aicrowd-production.s3.eu-central-1.amazonaws.com/dataset_files/challenge_287/1fe720be-90e4-4e06-9b52-9de93e0ea937_train.csv
# !wget https://aicrowd-production.s3.eu-central-1.amazonaws.com/dataset_files/challenge_287/fcac6286-6db1-4577-ad80-612fb9d36db9_test.csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_file = '/content/1fe720be-90e4-4e06-9b52-9de93e0ea937_train.csv'
df = pd.read_csv(data_file)
test_file = '/content/fcac6286-6db1-4577-ad80-612fb9d36db9_test.csv'
df1 = pd.read_csv(test_file)

df.text = df.text.str.replace('https?://[A-Za-z0-9./]+', '')
df.text = df.text.str.replace('@[a-zA-Z0-9_]+', '')
df.text = df.text.str.replace('#[a-zA-Z0-9_]+', '')
df.text = df.text.str.replace('&amp;', 'and')
df.text = df.text.str.lower()
df.text = df.text.str.replace("[']", '')
df.text = df.text.str.replace('[^a-zA-Z]', ' ')


df1.text = df1.text.str.replace('https?://[A-Za-z0-9./]+', '')
df1.text = df1.text.str.replace('@[a-zA-Z0-9_]+', '')
df1.text = df1.text.str.replace('#[a-zA-Z0-9_]+', '')
df1.text = df1.text.str.replace('&amp;', 'and')
df1.text = df1.text.str.lower()
df1.text = df1.text.str.replace("[']", '')
df1.text = df1.text.str.replace('[^a-zA-Z]', ' ')

clean_text = []
for tweet in df['text']:
  clean_text.append(' '.join(tweet.split()))

clean_text1 = []
for tweet in df1['text']:
  clean_text1.append(' '.join(tweet.split()))

import tensorflow_hub as hub
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

X_train = embed(clean_text).numpy()
y_train = df['labels'].values
X_test = embed(clean_text1).numpy()

from sklearn.svm import SVC
clf = SVC(C=28, gamma='auto')
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

predictions

# with open('/content/submission.csv', 'w') as f:
#   f.write('labels\n')
#   for pred in predictions:
#     f.write(str(pred) + '\n')

with open('/content/submission.csv', 'w') as f:
  f.write(',labels\n')
  for i in range(len(predictions)):
    f.write(str(i) + ',' + str(predictions[i]) + '\n')

