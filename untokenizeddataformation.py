import json
import random
import math
import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader


class DataCreation():
    def __init__(self):
        self.DataList = json.load(open(r"manim_scene_dataset.json"))

        pass
    def converting_to_continuous_Text(self):
        data_list = self.DataList
        formatted_list = []
        for unformatted_data_item in data_list:

            formatted_list.append( "for visualization of the scene with description: " + unformatted_data_item['description']\
                + "and duration: "+ str(unformatted_data_item['duration']) + ", python and manim based code is the follwoing:\n" \
                    + unformatted_data_item['manim_code'])

        return formatted_list
    def verify_formatting(self):
        data_list = self.DataList
        unformatted_data_item = data_list[0]
        formatted_item = "for visualization of the scene with description: " + unformatted_data_item['description']\
                + "and duration: "+ str(unformatted_data_item['duration']) + ", python and manim based code is the follwoing:\n" \
                    + unformatted_data_item['manim_code']

        return formatted_item
    def save_formatted_data(self, formatted_data_list):
        json.dump(formatted_data_list, open(r"formatted_manim_scene_dataset.json", "w"))

class LMDataset(Dataset):
    def __init__(self, txt_list,tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        
        def tokenize_one_sample(txt):
            token_ids = self.tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
            for i in range(0, len(token_ids) - self.max_length, self.stride):
                input_chunk = token_ids[i:i + self.max_length]
                target_chunk = token_ids[i + 1: i + self.max_length + 1]
                self.input_ids.append(torch.tensor(input_chunk))
                self.target_ids.append(torch.tensor(target_chunk))
    
        for txt in txt_list:
            tokenize_one_sample(txt)
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
        
def LMDataloader(txt_list,batch_size, max_length, stride, shuffle = True, drop_list = True, num_workers = 0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = LMDataset(txt_list,tokenizer, max_length, stride)
    
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, drop_last = drop_list, num_workers = num_workers)
    return dataloader

def finalDataLoader(batch_size, max_length, stride, train_ratio = 0.8, shuffle = True, drop_list = True, num_workers = 0):
    try:
        formatted_dataset = json.load(open(r"formatted_manim_scene_dataset.json"))
    except Exception as e:
        print(f"formatted_manim_scene_dataset.json not found with error:{e}")
        formatted_dataset = DataCreation().converting_to_continuous_Text()
    print("formatted sample size: ", len(formatted_dataset))
    DataCreation().save_formatted_data(formatted_dataset)

    
    print(formatted_dataset[0])
    print(len(formatted_dataset))
    split_idx = int(train_ratio * len(formatted_dataset))
    train_data = formatted_dataset[:split_idx]
    val_data = formatted_dataset[split_idx:]
    traindataloader = LMDataloader(
        txt_list=train_data,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride
    )
    valdataloader = LMDataloader(
        txt_list=val_data,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride
    )
    return traindataloader, valdataloader
        

        
        
if __name__ == "__main__":

    batch_size = 8
    max_length = 4
    stride = 1
    
    vocab_size = 50257
    output_dim = 256
    context_length = 1024
    
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    
    dataloader,_ = finalDataLoader(batch_size, max_length, stride)
    print(type(dataloader))
    
    
    
    for batch in dataloader:
        x, y = batch

        token_embeddings = token_embedding_layer(x)
        pos_embeddings = pos_embedding_layer(torch.arange(max_length))

        input_embeddings = token_embeddings + pos_embeddings

        break
