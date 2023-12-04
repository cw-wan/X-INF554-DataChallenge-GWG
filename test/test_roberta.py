from transformers import RobertaModel, RobertaTokenizer
import os

os.chdir("../")

tokenizer = RobertaTokenizer.from_pretrained('modules/roberta-large/', local_files_only=True)
encoder = RobertaModel.from_pretrained('modules/roberta-large/', local_files_only=True)

encoder.eval()

texts = [
    "To be or not to be,",
    "This is a problem."
]

tokens = tokenizer(texts, return_tensors='pt', padding=True)
embeddings = encoder(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"])["last_hidden_state"]

print(embeddings.shape)
