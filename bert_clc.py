import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer
from transformers import BertModel
from collections import Counter, defaultdict
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import time
from dataset_loader import load_data
from torch.nn import functional as F


# ----------Hyperparameters of the entire pipeline----------
# --------------Claim Label Classification--------------
d_bert_base = 768
d_bert_large = 1024
gpu = 0
input_seq_max_len = 384
loader_batch_size = 24
loader_worker_num = 2
num_epoch = 7
num_of_classes = 3
opti_lr_clc = 2e-5
label_mapper_ltoi = {'SUPPORTS': 0, 'REFUTES': 1, 'NOT_ENOUGH_INFO': 2}
label_mapper_itol = {0: 'SUPPORTS', 1: 'REFUTES', 2: 'NOT_ENOUGH_INFO'}
# ------------------------------------------------------

class CFEVERLabelTrainDataset(Dataset):
    """Climate Fact Extraction and Verification Dataset for Training, for the Evidence Retrival task."""

    def __init__(self, claims, evidences_, max_len=input_seq_max_len):
        self.data_set = unroll_train_claim_evidence_pairs(claims)
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


def unroll_train_claim_evidence_pairs(claims):
    """
    Rule: 
    Current approach considers all evidences to be with the 
    label that the associated claim has, except for the DISPUTED label.
    """
    claim_evidence_pairs = []

    for claim_id in claims:
        if claims[claim_id]['claim_label'] != 'DISPUTED':
            for evidence_id in claims[claim_id]['evidences']:
                claim_evidence_pairs.append((claim_id, evidence_id, label_mapper_ltoi[claims[claim_id]['claim_label']]))
    
    return claim_evidence_pairs


class CFEVERLabelTestDataset(Dataset):
    """Climate Fact Extraction and Verification Dataset for Testing, for the Evidence Retrival task."""

    def __init__(self, claims, evidences_, max_len=input_seq_max_len):
        self.data_set = unroll_test_claim_evidence_pairs(claims)
        self.max_len = max_len
        self.claims = claims
        self.evidences = evidences_

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, index):
        claim_id, evidence_id = self.data_set[index]

        # Preprocessing the text to be suitable for BERT
        claim_evidence_in_tokens = self.tokenizer.encode_plus(self.claims[claim_id]['claim_text'], self.evidences[evidence_id], 
                                                              return_tensors='pt', padding='max_length', truncation=True,
                                                              max_length=self.max_len, return_token_type_ids=True)
        
        seq, attn_masks, segment_ids, position_ids = claim_evidence_in_tokens['input_ids'].squeeze(0), claim_evidence_in_tokens[
                'attention_mask'].squeeze(0), claim_evidence_in_tokens['token_type_ids'].squeeze(0), torch.tensor([i+1 for i in range(self.max_len)])
    
        return seq, attn_masks, segment_ids, position_ids, claim_id


def unroll_test_claim_evidence_pairs(claims):
    claim_evidence_pairs = []

    for claim_id in claims:
        for evidence_id in claims[claim_id]['evidences']:
            claim_evidence_pairs.append((claim_id, evidence_id))
    
    return claim_evidence_pairs


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


def train_claim_cls(net, loss_criterion, opti, train_loader, dev_loader, dev_claims, gpu, max_eps=num_epoch):
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
            loss = loss_criterion(logits, labels)

            # Backpropagating the gradients, account for gradients
            loss.backward()

            # Optimization step, apply the gradients
            opti.step()

            if i % 100 == 0:
                acc = get_accuracy_from_logits(logits, labels)
                print("Iteration {} of epoch {} complete. Loss: {}; Accuracy: {}; Time taken (s): {}".format(i, ep, loss.item(), acc, (time.time() - st)))
                st = time.time()

        dev_acc = evaluate_dev(net, dev_loader, dev_claims, gpu)
        print("\nEpoch {} complete! Development Accuracy on dev claim labels: {}.".format(ep, dev_acc))
        if dev_acc > best_acc:
            print("Best development accuracy improved from {} to {}, saving model...\n".format(best_acc, dev_acc))
            best_acc = dev_acc
            torch.save(net.state_dict(), '/content/drive/MyDrive/Colab Notebooks/Assignment3/cfeverlabelcls.dat')


def get_accuracy_from_logits(logits, labels):
    probs = F.softmax(logits, dim=-1)
    predicted_classes = torch.argmax(probs, dim=1)
    acc = (predicted_classes.squeeze() == labels).float().mean()
    return acc


def get_predictions_from_logits(logits):
    probs = F.softmax(logits, dim=-1)
    predicted_classes = torch.argmax(probs, dim=1)
    return predicted_classes.squeeze()


def predict_pairs(net, dataloader, gpu):
    net.eval()

    claim_evidence_labels = defaultdict(list)
    df = pd.DataFrame()

    with torch.no_grad():
        for seq, attn_masks, segment_ids, position_ids, claim_ids in dataloader:
            seq, attn_masks, segment_ids, position_ids = seq.cuda(gpu), attn_masks.cuda(gpu), segment_ids.cuda(gpu), position_ids.cuda(gpu)
            logits = net(seq, attn_masks, segment_ids, position_ids)
            preds = get_predictions_from_logits(logits)

            df = pd.concat([df, pd.DataFrame({"claim_ids": claim_ids, "preds": preds.cpu()})], ignore_index=True)


    for _, row in df.iterrows():
        claim_id = row['claim_ids']
        label = row['preds']

        claim_evidence_labels[claim_id].append(label)
    
    return claim_evidence_labels


def decide_claim_labels(net, dataloader, gpu):
    claim_evidence_labels = predict_pairs(net, dataloader, gpu)
    claim_labels = {}

    # for claim_id in claim_evidence_labels:
    #     if len(set(claim_evidence_labels[claim_id])) == 1:
    #         claim_labels[claim_id] = label_mapper_itol[claim_evidence_labels[claim_id][0]]
    #     elif len(set(claim_evidence_labels[claim_id])) == 2:
    #         if label_mapper_ltoi['NOT_ENOUGH_INFO'] in claim_evidence_labels[claim_id]:
    #             claim_labels[claim_id] = label_mapper_itol[(set(claim_evidence_labels[claim_id]) - {label_mapper_ltoi['NOT_ENOUGH_INFO']}).pop()]  # label as the other one: supports/refutes
    #         else:
    #             claim_labels[claim_id] = "DISPUTED"
    #     else:  # len(set(claim_evidence_labels[claim_id])) == 3
    #         claim_labels[claim_id] = "DISPUTED"

    for claim_id in claim_evidence_labels:
        claim_labels[claim_id] = label_mapper_itol[Counter(claim_evidence_labels[claim_id]).most_common(1)[0][0]]  # label as the most common one - majority voting
    
    return claim_labels


def evaluate_dev(net, dataloader, dev_claims, gpu):
    claim_labels = decide_claim_labels(net, dataloader, gpu)

    correct_labels = 0

    for claim_id in dev_claims:
        if claim_labels[claim_id] == dev_claims[claim_id]["claim_label"]:
            correct_labels += 1
    
    return correct_labels / len(dev_claims)  # claim label accuracy


def extract_claim_labels(test_claims, claim_labels):
    for claim in claim_labels:
        test_claims[claim]["claim_label"] = claim_labels[claim]
    
    with open('/content/drive/MyDrive/Colab Notebooks/Assignment3/test-claims-predictions.json', 'w') as f:
        json.dump(test_claims, f)
    
    return test_claims


if __name__ == '__main__':
    # train_claims, dev_claims, _, evidences = load_data()

    # net_clc = CFEVERLabelClassifier()
    # #net_clc.cuda(gpu) #Enable gpu support for the model

    # loss_criterion = nn.CrossEntropyLoss()
    # opti_clc = optim.Adam(net_clc.parameters(), lr=opti_lr_clc)

    # train_set = CFEVERLabelTrainDataset(train_claims, evidences)
    # dev_set = CFEVERLabelTestDataset(dev_claims, evidences)

    # train_loader = DataLoader(train_set, batch_size=loader_batch_size, num_workers=loader_worker_num)
    # dev_loader = DataLoader(dev_set, batch_size=loader_batch_size, num_workers=loader_worker_num)

    # train_claim_cls(net_clc, loss_criterion, opti_clc, train_loader, dev_loader, dev_claims, gpu)