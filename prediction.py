import pandas as pd
import numpy as np
import torch
import re
import pickle
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import BertForSequenceClassification, BertTokenizer
import argparse
import pandas as pd
import numpy as np
import os, subprocess, multiprocessing, json
import sys
import gc
import torch
import boto3
from urllib.parse import urlparse
import general_functions as gf
# import en_core_web_sm
# nlp = spacy.load("en_core_web_sm")


# import tarfile
# import pickle

def Prepare_Model(local_model_pkl_path):
    
    # def load_model(local_model_pkl_path):
        # with open(local_model_pkl_path, 'rb') as f:
        #     config, state_dict  = pickle.load(f)
        # return config, state_dict
    with open(local_model_pkl_path, "rb") as f:
        config, state_dict = pickle.load(f)
        # print(config)
    return config, state_dict 


def count_intnt_entits(text):
    doc = nlp(text)
    intents = [token.text for token in doc if token.pos_ == 'VERB']
    entities = [token.text for token in doc if token.pos_ in {'NOUN', 'PROPN', 'ADJ', 'NUM', 'ADV'}]
    return len(intents), len(entities)

def extract_ner_entities(sentence):
    doc = nlp(sentence)
    entities = [(entity.text, entity.label_) for entity in doc.ents]
    return entities

def length_entities(list_entities):
    if (list_entities==np.nan or list_entities==None or list_entities==''):
        return 0
    else:
        return len(list_entities)
    
def filter_named_entities(text):
    # Process the text using Spacy
    doc = nlp(text)
    # Filter out named entities (ORG, PERSON, and GPE tags)
    filtered_words = [token.text for token in doc if token.ent_type_ not in ['ORG', 'PERSON', 'GPE', "LOC", "FAC"]]
    # Join the filtered words back into a string
    filtered_text = ' '.join(filtered_words)
    return filtered_text

# def text_preprocess(col, list_2):
#     df = pd.DataFrame({ 'text': col })
#     df = df.drop_duplicates()
#     df[['no_of_intents', 'no_of_entities']] = df.apply(lambda x: pd.Series(count_intnt_entits(x['text'])), axis=1)  

#     df['ner_enities'] = ''
#     df.loc[df['text']!='', 'ner_enities'] = df.loc[df['text']!='', 'text'].apply(extract_ner_entities)
#     df['len_ner_enities'] = df['ner_enities'].apply(length_entities)
#     df3 = df[df['len_ner_enities']>0]
#     df3['text'] = df3['text'].apply(filter_named_entities)
#     df6 = pd.concat([df[df['len_ner_enities']==0], df3], axis = 0)
#     df6 = df6.drop(['no_of_intents','no_of_entities','ner_enities','len_ner_enities'], axis=1)

#     df6['text'] = df6['text'].str.strip()
    
#     return df6['text'].to_list()


def clean_text(text_list):
    # Clean the text
    # text_list = text_preprocess(text_list, list_2)
    text_list = [text for text in text_list if text.strip() and not    set(text).issubset(set(string.punctuation+string.whitespace))]
    text_list = [x.lower() for x in text_list]
    # Define a translation table to replace punctuation and special characters with empty string
    translator = str.maketrans(string.punctuation + "_", " " * len(string.punctuation + "_"))
    # Loop through each text in the list and clean it
    cleaned_list = []
    for text in text_list:
        # Replace punctuation and special characters with empty string
        cleaned_text = text.translate(translator)
        # Remove any remaining special characters, punctuation, or whitespaces
        cleaned_text = ' '.join(cleaned_text.split())
        cleaned_list.append(cleaned_text)
    
    return cleaned_list



def predict_ec_model(list_of_texts_new, category_name):
    # Clean the text
    def clean_text(text):
        clean_text = re.sub(r'[^\w\s]', '', text)
        clean_text1 = re.sub("(\d*\.\d+)|(\d+\.[0-9 ]+)","",clean_text)
        return clean_text1
    list_of_texts_new = [x.lower() for x in list_of_texts_new]
    cleaned_texts = [clean_text(text) for text in list_of_texts_new]
    # cleaned_texts = clean_text(list_of_texts_new)
    
    # Set the random seed for NumPy
    np.random.seed(42)

    # Set the random seed for PyTorch
    torch.manual_seed(42)

    # Load the model and configuration from the pickle file
       # Load the model and configuration from the pickle file
    # with open("CC_model.pkl", "rb") as f:
    #     config, state_dict = pickle.load(f)
    #     print(state_dict)

    # Initialize a new model object with the loaded configuration
    model = BertForSequenceClassification(model1[0])

    # Load the saved state dictionary into the model
    model.load_state_dict(model1[1])

    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # Prepare the input data
    encodings = tokenizer(cleaned_texts, truncation=True, padding=True)

    # Create the dataset
    dataset = torch.utils.data.TensorDataset(torch.tensor(encodings['input_ids']), torch.tensor(encodings['attention_mask']))

    # Create the DataLoader for the new data
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    predictions = []
    probabilities = []

    with torch.no_grad():
        model.eval()

        for batch in loader:
            input_ids, attention_mask = batch
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            probabilities.extend(probs[:, 1].tolist())
            predictions.extend(torch.argmax(logits, dim=1).tolist())

    # Create a new DataFrame with the predictions and probabilities
    new_data = pd.DataFrame()
    new_data['text'] = cleaned_texts
    new_data['predictions'] = pd.Series(predictions).apply(lambda x: category_name if x==1 else 'Other').values
    new_data['probability'] = probabilities
    # new_data.to_csv("ac.csv",index=False)

    return new_data

# df =pd.read_excel("CC_unseen_data_v5.xlsx")



# print(list_of_text)
# list_of_texts = ["HmCstmElderCarePlusLandingPageOpen", "HmCstmAdultCarePlusLandingPageOpen", "HmCstmgrandparentsCarePlusLandingPageOpen", "elderly care plus",
#                 "Daycare expens reimbursement", "baby care licensed","contentPage 2023 Eldercare Subsidy","contentPage 2023 Elder care Subsidy", 
#                  "Elderly Care Plus Information"]


def Batch_Inferencing(df_new, temp_dir, model, savefile=False):
     
    NB_BATCHES = 2
    len_batch = len(df_new)//NB_BATCHES
    

    
    generate_df = (df_new.iloc[i*len_batch:(i+1)*len_batch] for i in range(NB_BATCHES+1))
    # print("generate_df",generate_df)
    if not os.path.isdir(temp_dir):
        os.mkdir(temp_dir)
     
    for j,df_new in enumerate(generate_df): 
        # print("sub_df",df_new)
        # print("sub_df_neww",j)
        if df_new.empty:
            continue
        df_out  = predict_ec_model(df1['text_cleaned_li'], category_name='Child care')
        # print("generate_df",df_out.shape)
        # if drop_features:
        #     df_out = df_out.loc[:,["prediction_probabilities", "prediction"]]
        df_out.to_excel(os.path.join(temp_dir, f"Batch_{j}.xlsx"), index=False)
        print(f"==================Completed Model Inferencing for Batch No. {j}======================")
    
    Final_Preds = (
        pd.concat([pd.read_excel(os.path.join(temp_dir, file)) for file in os.listdir(temp_dir) if               file.endswith(".xlsx")]).reset_index(drop=True)
    )
    # print("aavv",Final_Preds)
    output_filename = f"{temp_dir}/Final_Preds_{gf.file_time()}.xlsx" 
    # print("ndjk",output_filename)
    # if savefile:                                                       
    Final_Preds.to_excel(output_filename, index=False)
        # return output_filename
    return output_filename, Final_Preds



def s3_file_operation(s3_uri, op_type="read", upload_filename=None):
    """
    Reads/Downloads files from s3. Uploads local files to s3.
    
    Params:
    
    s3_uri: Full s3 uri for any operation
    op_type: String "read"/"download"/"upload" based on the operation type
    upload_filename: file to be uploaded from local. Only applicable when op_type is set to "upload"
    
    Returns:
    
    Only returns the last part (delimited by "/") of the s3 object key when op_type = "read"/"download".
    The relative path can then be passed on to next lines.
    
    Returns the full s3 path when op_type="upload"
    """
    parsed_uri = urlparse(s3_uri, allow_fragments=False)
    bucket = parsed_uri.netloc
    key = parsed_uri.path[1:]
    s3_client = boto3.client('s3', region_name='us-east-1')
    if op_type == "read":
        fileobj = s3_client.get_object(Bucket=bucket, Key=key)     
        filedata = fileobj['Body'].read()
        # contents = filedata.decode('utf-8')
        return filedata
    elif op_type == "download":
        s3_client.download_file(bucket, key, key.split("/")[-1])
        return key.split("/")[-1]
    elif op_type == "upload":
        if upload_filename is None:
            raise ValueError("Specify File to be upload from local file system")
        s3_client.upload_file(upload_filename, bucket , key+'/'+upload_filename)
        return f"s3://{bucket}/{key}/{upload_filename}"


def updatedf(rawdf, predf, recoverlist):
    return pd.concat([rawdf[recoverlist],predf], axis=1)
    
def parse_arg():
    """
    This function parses command line arguments to this script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_s3_uri", type=str, required=True)
    parser.add_argument("--model_config", type=str, required=True)
    params = vars(parser.parse_args())

    return params


if __name__ == "__main__":
    
    print("Parsing command line arguments......\n")
    params = parse_arg()  
    data_path = params["file_s3_uri"]
    config_uri = params["model_config"]

    print("Downloading Model Configurations From s3......\n") 
    model_config = json.loads(s3_file_operation(config_uri))
    model_path = model_config["Model_Path"]
    model_local_copy = s3_file_operation(model_path, op_type="download")
    # print("sss",model_local_copy)
#     # model_local_copy = unzip_tar_gz(model_local_copy, "Trained_Model")

    print(f"Preparing Model......\n")
    model1 = Prepare_Model(model_local_copy)
    # print("aaa",model1)
    # print("aaa",model1)
    

    print("Loading data......\n")
    df = pd.read_csv(data_path)
    columns = ['client_id', 'person_internal_id', 'input','unit_name','category' ]
    df1 = pd.DataFrame(df, columns=columns)
    df1 = df1[df1['input'].notna()]
    # print(df['text_cleaned'].isnull().sum())
  
    df1['text_cleaned_li'] = df1['input'].tolist()
    # print(df.head(2))
    # print(df['text_cleaned_li'].dtypes)
    # print("asdfg",df_new)
    # list_of_texts = df_new
    # print('aa',list_of_texts)
    df_copy = df1.copy()
    
#     if "recover_list" in model_config:
#         full_df.drop(model_config["recover_list"], axis=1, inplace=True)

    print("Inferencing Started.....\n")
    pred_dir = "Predictions_dir"
        
        
        
    local_output_preds_file,outputdf = Batch_Inferencing(df1, pred_dir, model1)
    # print('hhahah',local_output_preds_file)
    print('sdsdgsdgs',outputdf.shape)
    
#     local_output_preds_file, outputdf = Batch_Inferencing(full_df, pred_dir, model)
    

    if "recover_list" in model_config:
        outputdf = updatedf(df_copy,outputdf, model_config["recover_list"])
    # outputdf = outputdf.assign(probability_labels=[np.array(["other","child care"])]*len(outputdf))
    outputdf.to_excel(local_output_preds_file, index=False)

    print("Copy to s3 Started......\n")
    output_s3_key_uri = model_config["Output_File_Path"]
    output_s3_uri = s3_file_operation(output_s3_key_uri, op_type="upload", upload_filename= local_output_preds_file)

    print("Predictions Saved.......!!\n")
    