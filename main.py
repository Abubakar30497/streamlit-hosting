import streamlit as st
import nltk
nltk.download('punkt')
nltk.download("stopwords")
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import requests
from bs4 import BeautifulSoup 
import os
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from PIL import Image
from transformers import BartTokenizer, BartForConditionalGeneration, AutoTokenizer
from transformers import LEDTokenizer, LEDForConditionalGeneration
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import clip

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")
model = AutoModelForSeq2SeqLM.from_pretrained("allenai/led-base-16384", gradient_checkpointing=True)

def summarize(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=2000, truncation=True).to(device)
    summary_ids = model.generate(inputs['input_ids'], num_beams=4,min_length = 300, max_length=500, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
    
def group_documents(documents, labels):
    grouped_documents = defaultdict(list)
    for doc, label in zip(documents, labels):
        grouped_documents[label].append(doc)
    result = []
    for label in sorted(grouped_documents.keys()):
        result.append('\n\n'.join(grouped_documents[label]))
    return result
def extract_content(url, length_limit=100):
    # Make a GET request to the URL
    response = requests.get(url)

    # Use BeautifulSoup to parse the HTML content of the page
    soup = BeautifulSoup(response.content, "html.parser")

    # Find all the paragraphs in the page
    paragraphs = soup.find_all("p")

    # Filter the paragraphs based on the length limit
    filtered_paragraphs = [p.text for p in paragraphs if len(p.text) > length_limit and p.text.strip() != '']

    # Concatenate all the filtered paragraphs into a single string
    combined_text = " ".join(filtered_paragraphs)

    # Remove escape sequences and any words in all capital letters
    combined_text = combined_text.replace("\n", " ").replace("\r", " ")
    combined_text = ' '.join([word for word in combined_text.split() if not (word.isupper() and word.isalpha())])

    # Trim spaces between two words if there are more than one at a time
    combined_text = ' '.join(combined_text.split())

    return combined_text

def textscrapper(t):
  # Input capture
  #text = input("Enter a query: ")
  text = t
  
  # Text preprocessing
  text = text.lower() # convert to lowercase
  text = re.sub(r'[^\w\s]','',text) # remove punctuations
  
  # Tokenization
  tokens = word_tokenize(text)
  
  # Stop word removal
  stop_words = set(stopwords.words("english"))
  tokens = [word for word in tokens if word not in stop_words]
  
  # Stemming
  #ps = PorterStemmer()
  #tokens = [ps.stem(word) for word in tokens]
  
  # lemmatizing
  #le = WordNetLemmatizer()
  #tokens = [le.lemmatize(word) for word in tokens]
  
  # Keyword extraction
  keywords = [word for word in tokens if len(word) > 1]
  
  # Output
  print("Preprocessed Tokens:", tokens)
  print("Keywords:", keywords)

  # your API key
  API_KEY = 'AIzaSyB8xsHozllAG2pG9f_Sz1yePy6EZe_d_qg'
  
  # your custom search engine ID
  CSE_ID = '834a8c24af6024f65'
  
  # Define the API endpoint
  url = "https://www.googleapis.com/customsearch/v1?q=QUERY&key=API_KEY&cx=CSE_ID"
  
  query = " ".join(keywords)
  # Replace QUERY, API_KEY, and CSE_ID with the appropriate values
  url = url.replace("QUERY", query)
  url = url.replace("API_KEY", API_KEY)
  url = url.replace("CSE_ID", CSE_ID)
  
  # Make the API request
  response = requests.get(url)
  
  # Parse the JSON response
  data = response.json()
  
  # Extract the search results
  results = data["items"]
  for result in results:
      title = result["title"]
      link = result["link"]
      print(f"Title: {title}")
      print(f"Link: {link}")
  
  # Create a list to store the information
  rows = []
  
  # Extract the link from the API response
  results = response.json()
  
  for result in results['items']:
      link = result['link']
      timeout = 60
      print(link)
      # Send a request to the webpage
      #page = requests.get(link)
      # Send a request to the webpage with a timeout
      try:
          page = requests.get(link, timeout=timeout)
      except Timeout:
          print(f'Timed out while fetching {link}, skipping')
          continue
  
      # Parse the HTML content of the page using BeautifulSoup
      soup = BeautifulSoup(page.text, 'html.parser')
      #soup2 = BeautifulSoup(page.content, "html.parser")
  
      # Extract the title of the page
      try:
        title = soup.find('title').get_text()
      except:
        continue
      # Remove script and style tags
      for script in soup(["script", "style"]):
          script.extract()
      # Extract the text of the page
      text = extract_content(link, length_limit=70)
      #text = soup.get_text()
  
      # Remove any extra whitespace
      text = re.sub('\s+', ' ', text)
      # A list of special_characters to be removed
      special_characters=['@','#','$','*','&','/','{','}','[',']','(',')','^','\\','|','<','>','~','`','+','_','-','=']
      #text.append(page_text)
      for i in special_characters:
      # Replace the special character with an empty string
        text=text.replace(i,"")
      # Extract the images from the page
      images = []
      for img in soup.find_all('img'):
          src = img.get('src')
          images.append(src)
  
      # Store the information in a list
      row = [title, text, images]
      #print(row)
      # Add the list to the list of rows
      rows.append(row)
  
  # Create a dataframe from the list of rows
  df = pd.DataFrame(rows, columns=['title', 'text', 'images'])
  
  # Droping irrelevent rows
  df = df[~df['text'].str.contains('donation')]
  df = df[~df['text'].str.contains('do not have access')]
  df = df[~df['text'].str.contains('Sorry! Something went wrong!')]
  df = df[~df['text'].str.contains('All rights reserved')]
  df = df[~df['text'].str.contains('SUBSCRIBE!')]
  df = df[~df['text'].str.contains('403 Forbidden')]
  df = df[~df['text'].str.contains('Access denied')]
  df = df[~df['text'].str.contains('Just a moment')]
  df = df[~df['text'].str.contains('Become a member')]
  df = df[~df['text'].str.contains('sexual harassment')]
  df = df[~df['text'].str.contains('download')]
  df = df[~df['text'].str.contains('get free access')]
  df = df[~df['text'].str.contains('Sign up')]
  
  
  df = df[~df['title'].str.contains('donation')]
  df = df[~df['title'].str.contains('do not have access')]
  df = df[~df['title'].str.contains('Sorry! Something went wrong!')]
  df = df[~df['title'].str.contains('All rights reserved')]
  df = df[~df['title'].str.contains('SUBSCRIBE!')]
  df = df[~df['title'].str.contains('403 Forbidden')]
  df = df[~df['title'].str.contains('Access denied')]
  df = df[~df['title'].str.contains('Just a moment')]
  df = df[~df['title'].str.contains('Become a member')]
  df = df[~df['title'].str.contains('sexual harassment')]
  df = df[~df['title'].str.contains('download')]
  df = df[~df['title'].str.contains('get free access')]
  df = df[~df['title'].str.contains('Sign up')]
  
  
  df.replace('', np.nan, inplace=True)
  df = df.dropna()
  
  # reseting index
  df = df.reset_index(drop = True)
  
  documents = []
  for text in df['text']:
    documents.append(text)


  
  
  # Compute TF-IDF features
  vectorizer = TfidfVectorizer()
  X = vectorizer.fit_transform(documents)
  
  # Cluster documents with KMeans
  n_clusters = 1
  if X.shape[0]<3:
    n_clusters = X.shape[0]  
  kmeans = KMeans(n_clusters=n_clusters)
  kmeans.fit(X)
  
  # Reduce dimensionality with PCA
  pca = PCA(n_components=2)
  X_pca = pca.fit_transform(X.toarray())
  
  # Plot clusters
  colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
  plt.figure(figsize=(8, 8))
  for i in range(n_clusters):
      plt.scatter(X_pca[kmeans.labels_ == i, 0], X_pca[kmeans.labels_ == i, 1], c=colors[i], label='Cluster '+str(i+1))
  plt.legend()
  plt.show()
  
  #get the cluster labels
  labels = kmeans.labels_
  
  # create a dataframe with the documents and their assigned cluster labels
  df = pd.DataFrame({'document': documents, 'cluster': labels})
  
  # print the dataframe
  print(df)


  grouped_docs = group_documents(documents, labels)
  #print(grouped_docs[0])
  
  return grouped_docs



# Streamlit code starts here

st.title("Text Analysis Tool")

# Create a text input field
query_text = st.text_input("Enter a query:")

# Create a button to trigger analysis
#if st.button("Fetch and Analyze"):
    # Perform analysis when the button is clicked
extracted_text = textscrapper(query_text)
    
    # Display extracted and clustered text
#st.subheader("Extracted and Clustered Text:")
#st.write(extracted_text)

st.subheader("Summary:")
sums = []
for text in extracted_text[::-1]:
    sums.append(summarize(text))
mediate_summary = ''
for s in sums:
  mediate_summary+=' '+s
    
st.write(summarize(mediate_summary))
# Streamlit code ends here

# Rest of your code
# ...






# Function to display entered text in text box
def display_text():
    #text = input_field.get()
    text = textscrapper(input_field.get())
  
    text_box.insert(END, text + "\n")




# Input field
Label(root, text="Enter text:").pack()
input_field = Entry(root)
input_field.pack()



# Text box
Label(root, text="Text box:").pack()
text_box = Text(root, height=5)
text_box.pack()

# Canvas for images
Label(root, text="Image display:").pack()
canvas = Canvas(root, width=300, height=300)
canvas.pack()



root.mainloop()

