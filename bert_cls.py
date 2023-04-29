import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random
from transformers import BertTokenizer
from transformers import BertModel
from collections import Counter, defaultdict
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import time
import copy
from dataset_loader import load_data
from torch.nn import functional as F


# ----------Hyperparameters of the entire pipeline----------
# --------------Claim Label Classification--------------
d_bert_base = 768
d_bert_large = 1024
gpu = 0
input_seq_max_len = 384
loader_batch_size = 32
loader_worker_num = 2
num_epoch = 1
num_of_classes = 3
# ------------------------------------------------------


class CFEVERLabelDataset(Dataset):
    """Climate Fact Extraction and Verification Dataset for Training, for the Evidence Retrival task."""

    def __init__(self, claims, evidences_, max_len=input_seq_max_len):
        self.data_set = unroll_claim_labels(claims)
        self.max_len = max_len
        self.claims = claims
        self.evidences = evidences_

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, index):
        claim_id, evidence_id, label = self.data_set[index]

        # Preprocessing the text to be suitable for BERT
        claim_evidence_in_tokens = self.tokenizer.encode_plus(self.claims[claim_id]['claim_text'], self.evidences[evidence_id], 
                                                              return_tensors='pt', padding='max_length', truncation=True,
                                                              max_length=self.max_len, return_token_type_ids=True)
        
        seq, attn_masks, segment_ids, position_ids = claim_evidence_in_tokens['input_ids'].squeeze(0), claim_evidence_in_tokens[
                'attention_mask'].squeeze(0), claim_evidence_in_tokens['token_type_ids'].squeeze(0), torch.tensor([i+1 for i in range(self.max_len)])
    
        return seq, attn_masks, segment_ids, position_ids, label


def unroll_claim_labels(claims):
    pass


class CFEVERLabelClassifier(nn.Module):
    def __init__(self):
        super(CFEVERLabelClassifier, self).__init__()

        # Instantiating BERT model object
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Classification layer
        # input dimension is 768 because [CLS] embedding has a dimension of 768, if bert base is used
        # output dimension is 1 because we're working with a binary classification problem - RELEVANT : NOT RELEVANT
        self.cls_layer = nn.Linear(d_bert_base, num_of_classes)

    def forward(self, seq, attn_masks, segment_ids, position_ids):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
            -segment_ids : Tensor of shape [B, T] containing token ids of segment embeddings (see BERT paper for more details)
            -position_ids : Tensor of shape [B, T] containing token ids of position embeddings (see BERT paper for more details)
        '''
        
        # Feeding the input to BERT model to obtain contextualized representations
        outputs = self.bert(seq, attention_mask=attn_masks, token_type_ids=segment_ids, position_ids=position_ids, return_dict=True)
        cont_reps = outputs.last_hidden_state

        # Obtaining the representation of [CLS] head (the first token)
        cls_rep = cont_reps[:, 0]

        # Feeding cls_rep to the classifier layer
        logits = self.cls_layer(cls_rep)

        return logits  # logits shape is [B, num_of_classes]


def train_claim_cls(net, loss_criterion, opti, train_loader, dev_loader, gpu, max_eps=num_epoch):
    best_acc = 0
    st = time.time()

    for ep in range(max_eps):
        net.train()  # Good practice to set the mode of the model
        
        for i, (seq, attn_masks, segment_ids, position_ids, labels) in enumerate(train_loader):
            # Reset/Clear gradients
            opti.zero_grad()

            # Extracting the tokens ids, attention masks and token type ids
            seq, attn_masks, segment_ids, position_ids, labels = seq.cuda(gpu), attn_masks.cuda(gpu), segment_ids.cuda(gpu), position_ids.cuda(gpu), labels.cuda(gpu)

            # Obtaining the logits from the model
            logits = net(seq, attn_masks, segment_ids, position_ids)

            # Computing loss
            loss = loss_criterion(logits, labels.float())

            # Backpropagating the gradients, account for gradients
            loss.backward()

            # Optimization step, apply the gradients
            opti.step()

            if i % 100 == 0:
                acc = get_accuracy_from_logits(logits, labels)
                print("Iteration {} of epoch {} complete. Loss: {}; Accuracy: {}; Time taken (s): {}".format(i, ep, loss.item(), acc, (time.time() - st)))
                st = time.time()

        dev_acc = evaluate(net, dev_loader, gpu)
        print("Epoch {} complete! Development F1: {}; Development Accuracy:{}".format(ep, dev_acc))
        if acc > best_acc:
            print("Best development f1 improved from {} to {}, saving model...".format(best_acc, dev_acc))
            best_acc = dev_acc
            torch.save(net.state_dict(), 'cfeverercls_{}.dat'.format(ep))


def get_accuracy_from_logits(logits, labels):
    probs = F.softmax(logits, dim=-1)
    predicted_classes = torch.argmax(probs)
    acc = (predicted_classes.squeeze() == labels).float().mean()
    return acc


def evaluate(net, dataloader, gpu):
    net.eval()

    mean_acc = 0
    count = 0

    with torch.no_grad():
        for seq, attn_masks, labels in dataloader:
            seq, attn_masks, labels = seq.cuda(gpu), attn_masks.cuda(gpu), labels.cuda(gpu)
            logits = net(seq, attn_masks)
            mean_acc += get_accuracy_from_logits(logits, labels)
            count += 1

    return mean_acc / count


if __name__ == '__main__':
    random.seed(42)
    train_claims, dev_claims, test_claims, evidences = load_data()

    # net = CFEVERLabelClassifier()
    # net.cuda(gpu) #Enable gpu support for the model

    # loss_criterion = nn.CrossEntropyLoss()
    # opti = optim.Adam(net.parameters(), lr=2e-5)


    # train_set = CFEVERLabelDataset(train_claims, evidences)
    # dev_set = CFEVERLabelDataset(dev_claims, evidences)
    # #test_set = CFEVERERDataset(test_claims, evidences)

    # train_loader = DataLoader(train_set, batch_size=loader_batch_size, num_workers=loader_worker_num)
    # dev_loader = DataLoader(dev_set, batch_size=loader_batch_size, num_workers=loader_worker_num)
    # #test_loader = DataLoader(test_set, batch_size=loader_batch_size, num_workers=loader_worker_num)


    # train_claim_cls(net, loss_criterion, opti, train_loader, dev_loader, dev_claims, gpu)