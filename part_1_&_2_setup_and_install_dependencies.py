# -*- coding: utf-8 -*-
!pip install transformers torchvision torchaudio --quiet
!pip install opencv-python moviepy --quiet
!pip install lxml_html_clean --quiet
!pip install newspaper3k --upgrade --quiet
!pip install instaloader --quiet
!pip install -q langchain langchain-community openai

from google.colab import files
drive.mount('/content/drive')


from newspaper import Article
from bs4 import BeautifulSoup
import requests
import instaloader
import re
import pandas as pd

df = pd.read_csv('/content/drive/MyDrive/detectiondataset.csv', encoding='utf-8')
df.head()
 # Add is_news? based on first 4 characters of 'id' column
df['is_news?'] = df['id'].apply(lambda x: 'real' if str(x).lower().startswith('real') else 'fake')
df['text_length'] = df['text'].apply(lambda x: len(str(x).split()))

#  drop the movies coulmn
df = df.drop(columns=['movies'], errors='ignore')
df = df.drop(columns=['publish_date'], errors='ignore')

# drop the rows where url column has null vales
df.dropna(subset=['url'], inplace=True)

# Fake_1-Webpage
# 1    Fake_10-Webpage
# 2    Fake_11-Webpage
# 3    Fake_12-Webpage
# 4    Fake_13-Webpage
# 5    Fake_14-Webpage
# 6    Fake_15-Webpage
# 7    Fake_16-Webpage
# 8    Fake_17-Webpage
# 9    Fake_18-Webpage
# remove the FAKE_  and -Webpage from the id's column

df['id'] = df['id'].astype(str).str.replace('Fake_', '').str.replace('Real_', '').str.replace('-Webpage', '').str.replace('-News', '')
print(df['id'].head(10))

print(df['is_news?'].apply(lambda x: str(x).lower().startswith('real')).sum(), "real IDs")
print(df['is_news?'].apply(lambda x: str(x).lower().startswith('fake')).sum(), "fake IDs")
df['image_path'] = df['image_path'].fillna('') if 'image_path' in df.columns else ''
df['video_path'] = df['video_path'].fillna('') if 'video_path' in df.columns else ''

# Recalculate number of words in each news text
df['text_length'] = df['text'].apply(lambda x: len(str(x).split()))

# Step 3: Display how many 'real' and 'fake' samples are present
print("Label distribution:\n", df['is_news?'].value_counts())

# Step 4: Visualize the distribution of real vs fake news
import matplotlib.pyplot as plt

df['is_news?'].value_counts().plot(kind='bar', title='News Real or Fake Distribution', xlabel='Label', ylabel='Count')
plt.tight_layout()
plt.show()

def extract_text_from_url(url):
    try:
        if 'youtube.com' in url:
            return "YouTube video title or transcript parsing not implemented."
        elif 'instagram.com' in url:
            shortcode = re.search(r'/p/([^/]+)/', url)
            if shortcode:
                shortcode = shortcode.group(1)
                try:
                    L = instaloader.Instaloader()
                    post = instaloader.Post.from_shortcode(L.context, shortcode)
                    return post.caption if post.caption else ""
                except:
                    try:
                        page = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
                        soup = BeautifulSoup(page.content, 'html.parser')
                        meta = soup.find('meta', property='og:description')
                        return meta['content'] if meta else ""
                    except:
                        return ""
        else:
            article = Article(url)
            article.download()
            article.parse()
            return article.text
    except:
        return ""

# Append new Instagram links (fake/real examples) manually
new_samples = [
    {"text": "https://www.instagram.com/p/DKi1TU2OXdr/", "image_path": "", "video_path": "", "label": "fake"},
    {"text": "https://www.instagram.com/p/DKfGCciRUGN/", "image_path": "", "video_path": "", "label": "fake"},
    {"text": "https://www.instagram.com/p/DKR8fBnRW9y/", "image_path": "", "video_path": "", "label": "fake"},
    {"text": "https://www.instagram.com/p/CKQzKtGjxu7/", "image_path": "", "video_path": "", "label": "real"},
    {"text": "https://www.bbc.com/news/articles/cly12egqq5ko","image_path": "", "video_path": "", "label": "fake"}
]
new_df = pd.DataFrame(new_samples)

for i in range(len(new_df)):
    new_df.loc[i, 'text'] = new_df.loc[i, 'text'].strip()
    try:
        extracted = extract_text_from_url(new_df.loc[i, 'text'])
        new_df.loc[i, 'text'] = extracted
    except:
        pass

# Combine with existing dataframe
try:
    df  # Check if df already exists
except NameError:
    df = pd.DataFrame(columns=["text", "image_path", "video_path", "label"])

df = pd.concat([df, new_df], ignore_index=True)

# ✅ Clean missing image and video paths
df['image_path'] = df['image_path'].fillna('')
df['video_path'] = df['video_path'].fillna('')
