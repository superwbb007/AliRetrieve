# from transformers import BertConfig, BertModel


# configuration = BertConfig()
# model  = BertModel(configuration)

# configuration = model.config

# from transformers import TFBertTokenizer
# tf_tokenizer = TFBertTokenizer.from_pretrained('bert-base-uncased')

# from_tokenizer
# tokenizer = 

from transformers import AutoTokenizer, BertModel
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

inputs = tokenizer('Hello, my dog is cute', return_tensors = 'pt')
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state

# tf_tokenizer = TFBertTokenizer.from_tokenizer(tokenizer)

# import torch
# tokenizer = AutoTokenizer.from