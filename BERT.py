import torch   
device = torch.device("cuda")

# !pip install transformers

import pandas as pd

df = pd.read_csv("/content/drive/My Drive/Colab Notebooks/fast.ai/train.csv")
df.sample(10)

df['text'] = df['text'].str.lower()
df['text'] = df['text'].str.replace('http\S+|www.\S+', '', case=False)
df['text'] = df['text'].str.replace('[^a-zA-Z]', ' ')

"""# Remove stopwords"""

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('stopwords')

df['text'] = df['text'].apply(lambda text : ' '.join([word for word in text.split() if not word in set(stopwords.words('english'))]))

sentences = df.text.values
labels = df.labels.values

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

print(' Original: ', sentences[0])
print('Tokenized: ', tokenizer.tokenize(sentences[0]))
print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0])))

"""**`[SEP]`**

At the end of every sentence, we need to append the special `[SEP]` token. 

This token is an artifact of two-sentence tasks, where BERT is given two separate sentences and asked to determine something (e.g., can the answer to the question in sentence A be found in sentence B?). 

I am not certain yet why the token is still required when we have only single-sentence input, but it is!

**`[CLS]`**

For classification tasks, we must prepend the special `[CLS]` token to the beginning of every sentence.

This token has special significance. BERT consists of 12 Transformer layers. Each transformer takes in a list of token embeddings, and produces the same number of embeddings on the output (but with the feature values changed, of course!).
"""

max_len = 0

for sent in sentences:
    input_ids = tokenizer.encode(sent, add_special_tokens=True)
    max_len = max(max_len, len(input_ids))
print('Max sentence length: ', max_len)

input_ids = []
attention_masks = []

for sent in sentences:
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 64,          # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,  # Construct attn. masks.
                        return_tensors = 'pt',         # Return pytorch tensors.
                   )
    
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids, attention_masks, labels = torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0), torch.tensor(labels)

print('Original: ', sentences[0])
print('Token IDs:', input_ids[0])

from torch.utils.data import TensorDataset, random_split

dataset = TensorDataset(input_ids, attention_masks, labels)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('%d training samples'%(train_size))
print('%d validation samples'%(val_size))

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

batch_size = 32

train_dataloader = DataLoader(
            train_dataset,  
            sampler = RandomSampler(train_dataset), 
            batch_size = batch_size
        )

validation_dataloader = DataLoader(
            val_dataset, 
            sampler = SequentialSampler(val_dataset), 
            batch_size = batch_size 
        )

from transformers import BertForSequenceClassification, AdamW, BertConfig

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels = 2, # HOF or NOT
    output_attentions = False,
    output_hidden_states = False, 
)

model.cuda()

optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)

from transformers import get_linear_schedule_with_warmup

epochs = 10

total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

"""## Training Loop"""

import numpy as np

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

import random
from tqdm import tqdm

seed_val = 39

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

for epoch_i in range(0, epochs):
    
    model.train()

    with tqdm(total=len(train_dataloader), desc='Train Epoch %d/%d'%(epoch_i+1, epochs)) as progress:
        total_train_loss = 0
        for step, batch in enumerate(train_dataloader):

            b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            model.zero_grad()        

            loss, logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            scheduler.step()
            avg_train_loss = total_train_loss / (step+1)
        
            progress.set_postfix_str(s='Average train Loss %f'%avg_train_loss, refresh=True)

            progress.update()

    model.eval()

    with tqdm(total=len(validation_dataloader), desc='Val Epoch %d/%d'%(epoch_i+1, epochs)) as progress:
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0
        for step, batch in enumerate(validation_dataloader):
            
            b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            with torch.no_grad():        

                (loss, logits) = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                
            total_eval_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            total_eval_accuracy += flat_accuracy(logits, label_ids)

            avg_val_accuracy = total_eval_accuracy / (step+1)
            avg_val_loss = total_eval_loss / (step+1)

            progress.set_postfix_str(s="Accuracy: %f, Val Loss %f"%(avg_val_accuracy, avg_val_loss), refresh=True)
            progress.update()

import pandas as pd

df = pd.read_csv("/content/drive/My Drive/Colab Notebooks/fast.ai/q1finaltest.csv")

sentences = df.text.values
labels = [0]*len(df) #dummy

input_ids = []
attention_masks = []

for sent in sentences:
    encoded_dict = tokenizer.encode_plus( sent, add_special_tokens = True, max_length = 64, pad_to_max_length = True, return_attention_mask = True, return_tensors = 'pt',)

    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

batch_size = 32  

prediction_data = TensorDataset(input_ids, attention_masks, labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

model.eval()

predictions , true_labels = [], []

with tqdm(total=len(prediction_dataloader), desc='Predict Test') as progress:
    for batch in prediction_dataloader:

        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        predictions.append(logits)
        true_labels.append(label_ids)

        progress.update()

labels_final=[]
for preds in predictions:
    labels_final += list(np.argmax(preds, axis=1))

final = pd.DataFrame()
final['labels'] =labels_final