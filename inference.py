import os
import boto3
import json
from io import StringIO

import pandas as pd
import numpy as np
import torch
import re
import pickle
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import BertForSequenceClassification, BertTokenizer
import spacy
import en_core_web_sm
import string
nlp = spacy.load("en_core_web_sm")

###########
#Model Serving
###########

def clean_text(df,text_cols):
    # Create a new dataframe to hold the cleaned text columns
    cleaned_df = pd.DataFrame()
    
    # Define the list of stopwords
    stop_words = set(stopwords.words('english'))
    
    # Clean each text column and add it to the cleaned dataframe
    for text_col in text_cols:
        text_list = df[text_col].tolist()
        text_list = [str(text) for text in text_list]
        text_list = [text if text.strip() and not
                     set(text).issubset(set(string.punctuation + string.whitespace)) else '' 
                     for text in text_list]
        text_list = [x.lower() for x in text_list]
        translator = str.maketrans(string.punctuation + string.digits + "_", " " * len(
            string.punctuation + string.digits + "_"))
        cleaned_list = []
        for text in text_list:
            cleaned_text = text.translate(translator)
            cleaned_text = ' '.join(cleaned_text.split())
            cleaned_text = ' '.join([word for word in cleaned_text.split() if word not in stop_words])
            cleaned_list.append(cleaned_text)
        cleaned_df[text_col] = cleaned_list
    
    # Add the non-text columns to the cleaned dataframe
    for col in df.columns:
        if col not in text_cols:
            cleaned_df[col] = df[col]
    
    return cleaned_df


def load_from_list_to_tensor(cleaned_texts):
    np.random.seed(42)
    # Set the random seed for PyTorch
    torch.manual_seed(42)
    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # Prepare the input data
    encodings = tokenizer(cleaned_texts, truncation=True, padding=True)
    # Create the dataset
    dataset = torch.utils.data.TensorDataset(torch.tensor(encodings['input_ids']), torch.tensor(encodings['attention_mask']))
    # Create the DataLoader for the new data
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
    return loader
    
"""
Deserialize fitted model
"""
def model_fn(local_model_pkl_path):
    with open(local_model_pkl_path, "rb") as f:
        model = joblib.load(f)
    return model 


"""
input_fn
    request_body: The body of the request sent to the model.
    request_content_type: (string) specifies the format/variable type of the request
"""
def input_fn(input_data, content_type):
    
    if content_type == "text/csv":
        # Read the raw input data as CSV.
        buff = StringIO(input_data.strip())
        df = pd.read_csv(buff, sep=",") 
        list_of_texts = [x.lower() for x in df.input]
#     cleaned_texts = [clean_text(text) for text in list_of_texts]
        cleaned_texts, ind_ = clean_text(list_of_texts)
        loaded_tensors = load_from_list_to_tensor(cleaned_texts)
        return loaded_tensors, ind_
    else:
        raise ValueError("{} not supported by script!".format(content_type))

"""
predict_fn
    input_data: returned array from input_fn above
    model (pytorch) returned model loaded from model_fn above
"""
def predict_fn(input_object, model):
    
    sent_trans_model = SentenceTransformer('all-mpnet-base-v2')
    cleaned_concatenated_df = clean_text(new_df1,text_cols=[text_col])
    # classifier = joblib.load(model_path)
    
    # Encode the text samples from the Parquet file
    encodings = sent_trans_model.encode(new_df1[text_col].fillna(' ').tolist())

    # Make predictions for the encoded samples
    # predictions = loaded_model.predict(encodings)
    probabilities = model1.predict_proba(encodings)
    
    # Apply threshold and assign labels
    new_data = pd.DataFrame()
    new_data['predicted_label'] = np.where(probabilities[:, 0] > threshold, category_name_1, 'Other')
    new_data['prediction'] = new_data['predicted_label'].map({category_name_1:1, 'Other':0})
    
    new_data['pred_probab'] = probabilities[:, 0]

    return new_data

# Serialize the prediction result into the desired response content type
#def output_fn(predictions, response_content_type):
    # Create a new DataFrame with the predictions and probabilities
    #new_data = pd.DataFrame(predictions)
    #predictions['predictions'] = predictions["preds"].apply(lambda x: category_name if x==1 else 'Other')
    #new_data = predictions.to_dict(orient='list')
    #return new_data
    