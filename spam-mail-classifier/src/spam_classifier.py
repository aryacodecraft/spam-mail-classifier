import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']] # latin-1 instead of utf-8 because datset contains special characters
df.columns = ['label', 'text']

labels = []
for i in df['label']:
    if i == 'normal':
        labels.append(0)
    else:
        labels.append(1)
df['label'] = labels

def clean(msg):
    msg = msg.lower()
    msg = re.sub(r'[^a-zA-Z\s]', '', msg)
    msg = re.sub(r'\s+', ' ', msg).strip()
    return msg

df['text'] = df['text'].apply(clean)
original_texts = df['text']

cv = CountVectorizer()
X = cv.fit_transform(df['text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# ---the actual spam messages---
# for i in range(len(y_pred)):
#     if y_pred[i] == 1:
#         print("->" + original_texts.iloc[i])