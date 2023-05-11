# This file contains the code for the Claim Label Classification task: Data -> Training -> Classification Model

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
from torch.nn import functional as F
from main import path_prefix


clc_model_params_filename = path_prefix + 'cfeverlabelcls.dat'

# ----------Hyperparameters of the entire pipeline----------
# --------------Claim Label Classification--------------
d_bert_base = 768
gpu = 0
input_seq_max_len = 384
loader_batch_size = 24
loader_worker_num = 2
num_epoch = 9
max_evi_num = 5
num_of_standalone_classes = 3
num_of_classes = 4
opti_lr_clc = 2e-5
label_mapper_ltoi = {'SUPPORTS': 0, 'REFUTES': 1, 'NOT_ENOUGH_INFO': 2, 'DISPUTED': 3}
label_mapper_itol = {0: 'SUPPORTS', 1: 'REFUTES', 2: 'NOT_ENOUGH_INFO', 3: 'DISPUTED'}
# ------------------------------------------------------

class CFEVERLabelTrainDataset(Dataset):
    """Climate Fact Extraction and Verification Dataset for Training, for the Evidence Retrival task."""

    def __init__(self, claims, evidences_, max_len=input_seq_max_len):
        self.data_set = [claims[c] for c in claims]
        self.max_len = max_len
        self.claims = claims
        self.evidences = evidences_

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, index):
        claim = self.data_set[index]

        evi_masks = torch.tensor([1] * min(len(claim['evidences']), max_evi_num) + [0] * max(0, max_evi_num - len(claim['evidences'])))

        while len(claim['evidences']) < max_evi_num:
            claim['evidences'].append('')

        seqs, attn_masks, segment_ids = [], [], []

        for eid in claim['evidences'][:max_evi_num]:
            # Preprocessing the text to be suitable for BERT
            claim_evidence_in_tokens = self.tokenizer.encode_plus(claim['claim_text'], self.evidences[eid] if eid != '' else '<PAD>', 
                                                                return_tensors='pt', padding='max_length', truncation=True,
                                                                max_length=self.max_len, return_token_type_ids=True)
        
            seq, attn_mask, segment_id = claim_evidence_in_tokens['input_ids'].squeeze(0), claim_evidence_in_tokens[
                    'attention_mask'].squeeze(0), claim_evidence_in_tokens['token_type_ids'].squeeze(0)
            
            seqs.append(seq)
            attn_masks.append(attn_mask)
            segment_ids.append(segment_id)

        # Convert a list of tensors to a tensor of lists
        # Now batches of [evidence num, corresponding len]
        seqs = torch.stack(seqs)
        attn_masks = torch.stack(attn_masks)
        segment_ids = torch.stack(segment_ids)
    
        return seqs, attn_masks, segment_ids, evi_masks, label_mapper_ltoi[claim['claim_label']]


class CFEVERLabelTestDataset(Dataset):
    """Climate Fact Extraction and Verification Dataset for Testing, for the Evidence Retrival task."""

    def __init__(self, claims, evidences_, max_len=input_seq_max_len):
        self.data_set = [(c, claims[c]) for c in claims]
        self.max_len = max_len
        self.claims = claims
        self.evidences = evidences_

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, index):
        claim_id, claim = self.data_set[index]

        evi_masks = torch.tensor([1] * min(len(claim['evidences']), max_evi_num) + [0] * max(0, max_evi_num - len(claim['evidences'])))

        while len(claim['evidences']) < max_evi_num:
            claim['evidences'].append('')

        seqs, attn_masks, segment_ids = [], [], []

        for eid in claim['evidences']:
            # Preprocessing the text to be suitable for BERT
            claim_evidence_in_tokens = self.tokenizer.encode_plus(claim['claim_text'], self.evidences[eid] if eid != '' else '<PAD>', 
                                                                return_tensors='pt', padding='max_length', truncation=True,
                                                                max_length=self.max_len, return_token_type_ids=True)
            
            seq, attn_mask, segment_id = claim_evidence_in_tokens['input_ids'].squeeze(0), claim_evidence_in_tokens[
                    'attention_mask'].squeeze(0), claim_evidence_in_tokens['token_type_ids'].squeeze(0)
            
            seqs.append(seq)
            attn_masks.append(attn_mask)
            segment_ids.append(segment_id)

        # Convert a list of tensors to a tensor of lists
        # Now batches of [evidence num, corresponding len]
        seqs = torch.stack(seqs)
        attn_masks = torch.stack(attn_masks)
        segment_ids = torch.stack(segment_ids)
    
        return seqs, attn_masks, segment_ids, evi_masks, claim_id


class CFEVERLabelClassifier(nn.Module):
    def __init__(self):
        super(CFEVERLabelClassifier, self).__init__()

        # Instantiating BERT model object
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Classification layer
        # input dimension is 768 because [CLS] embedding has a dimension of 768, if bert base is used
        # output dimension is 1 because we're working with a binary classification problem - RELEVANT : NOT RELEVANT
        self.mid_layer = nn.Linear(d_bert_base, num_of_standalone_classes)

        self.cls_layer = nn.Linear(num_of_standalone_classes * max_evi_num, num_of_classes)

    def forward(self, seqs, attn_masks, segment_ids, evi_masks):
        B = seqs.shape[0]

        seqs = seqs.view(-1, seqs.shape[-1]) # [B * max_evi_num, seq_len]
        attn_masks = attn_masks.view(-1, attn_masks.shape[-1]) # [B * max_evi_num, seq_len]
        segment_ids = segment_ids.view(-1, segment_ids.shape[-1]) # [B * max_evi_num, seq_len]
        evi_masks = evi_masks.view(-1, 1) # [B * max_evi_num, 1]

        # Feeding the input to BERT model to obtain contextualized representations
        outputs = self.bert(seqs, attention_mask=attn_masks, token_type_ids=segment_ids, return_dict=True)
        cont_reps = outputs.last_hidden_state

        # Obtaining the representation of [CLS] head (the first token)
        cls_rep = cont_reps[:, 0]

        # Feeding cls_rep to the classifier layer
        mlogits = self.mid_layer(cls_rep)  # logits shape is [B * max_evi_num, num_of_standalone_classes]

        probs = F.softmax(mlogits, dim=1)  # probs shape is [B * max_evi_num, num_of_standalone_classes]

        masked_probs = probs * evi_masks  # masked_probs shape is [B * max_evi_num, num_of_standalone_classes]

        masked_probs = masked_probs.view(B, -1)

        logits = self.cls_layer(masked_probs)  # logits shape is [B, num_of_classes]

        return logits  # logits shape is [B, num_of_classes]


def train_claim_cls(net, loss_criterion, opti, train_loader, dev_loader, dev_claims, gpu, max_eps=num_epoch):
    best_acc = 0
    mean_losses = [0] * max_eps

    for ep in range(max_eps):
        net.train()  # Good practice to set the mode of the model
        st = time.time()
        train_acc = 0
        count = 0
        
        for i, (b_seqs, b_attn_masks, b_segment_ids, b_evi_masks, b_label) in enumerate(train_loader):
            # Reset/Clear gradients
            opti.zero_grad()

            # Extracting the tokens ids, attention masks and token type ids
            b_seqs, b_attn_masks, b_segment_ids, b_evi_masks, b_label = b_seqs.cuda(gpu), b_attn_masks.cuda(gpu), b_segment_ids.cuda(gpu), b_evi_masks.cuda(gpu), b_label.cuda(gpu)

            # Obtaining the logits from the model
            logits = net(b_seqs, b_attn_masks, b_segment_ids, b_evi_masks)

            # Computing loss
            loss = loss_criterion(logits, b_label)

            mean_losses[ep] += loss.item()
            count += 1
            train_acc += get_accuracy_from_logits(logits, b_label)

            # Backpropagating the gradients, account for gradients
            loss.backward()

            # Optimization step, apply the gradients
            opti.step()

            if i % 100 == 0:
                print("Iteration {} of epoch {} complete. Time taken (s): {}".format(i, ep, (time.time() - st)))
                st = time.time()
        
        mean_losses[ep] /= count
        print(f"Epoch {ep} completed. Loss: {mean_losses[ep]}, Accuracy: {train_acc / count}.\n")

        dev_acc = evaluate_dev(net, dev_loader, dev_claims, gpu)
        print("\nEpoch {} complete! Development Accuracy on dev claim labels: {}.".format(ep, dev_acc))
        if dev_acc > best_acc:
            print("Best development accuracy improved from {} to {}, saving model...\n".format(best_acc, dev_acc))
            best_acc = dev_acc
            torch.save(net.state_dict(), clc_model_params_filename)
        else:
            print()


def get_accuracy_from_logits(logits, labels):
    probs = F.softmax(logits, dim=-1)
    predicted_classes = torch.argmax(probs, dim=1)
    acc = (predicted_classes.squeeze() == labels).float().mean()
    return acc


def get_predictions_from_logits(logits):
    probs = F.softmax(logits, dim=-1)
    predicted_classes = torch.argmax(probs, dim=1)
    return predicted_classes.squeeze()


def predict(net, dataloader, gpu):
    net.eval()

    claim_labels = {}
    df = pd.DataFrame()

    with torch.no_grad():
        for b_seqs, b_attn_masks, b_segment_ids, b_evi_masks, b_claim_id in dataloader:
            b_seqs, b_attn_masks, b_segment_ids, b_evi_masks = b_seqs.cuda(gpu), b_attn_masks.cuda(gpu), b_segment_ids.cuda(gpu), b_evi_masks.cuda(gpu)
            logits = net(b_seqs, b_attn_masks, b_segment_ids, b_evi_masks)

            preds = get_predictions_from_logits(logits)
            df = pd.concat([df, pd.DataFrame({'claim_ids': b_claim_id, 'preds': preds.cpu()})], ignore_index=True)

    for _, row in df.iterrows():
        claim_id = row['claim_ids']
        label = row['preds']

        claim_labels[claim_id] = label_mapper_itol[label]
    
    return claim_labels


def evaluate_dev(net, dataloader, dev_claims, gpu):
    claim_labels = predict(net, dataloader, gpu)

    correct_labels = 0

    for claim_id in dev_claims:
        if claim_labels[claim_id] == dev_claims[claim_id]["claim_label"]:
            correct_labels += 1
    
    return correct_labels / len(dev_claims)  # claim label accuracy


def extract_claim_evi_labels(test_claims, claim_labels, output_filename):
    for claim in claim_labels:
        test_claims[claim]["claim_label"] = claim_labels[claim]
    
    with open(output_filename, 'w') as f:
        json.dump(test_claims, f)
    
    print("Final test claims predictions file ready.")
    
    return test_claims


def clc_pipeline(train_claims, dev_claims, evidences):
    net_clc = CFEVERLabelClassifier()
    net_clc.cuda(gpu) # Enable gpu support for the model

    class_counts = Counter([train_claims[claim]["claim_label"] for claim in train_claims])
    class_weights = torch.tensor([(sum(class_counts.values()) / class_counts[c]) for c in label_mapper_ltoi.keys()])
    loss_criterion = nn.CrossEntropyLoss(weight=class_weights).cuda(gpu)
    opti_clc = optim.Adam(net_clc.parameters(), lr=opti_lr_clc)

    train_set = CFEVERLabelTrainDataset(train_claims, evidences)
    dev_set = CFEVERLabelTestDataset(dev_claims, evidences)

    train_loader = DataLoader(train_set, batch_size=loader_batch_size, num_workers=loader_worker_num)
    dev_loader = DataLoader(dev_set, batch_size=loader_batch_size, num_workers=loader_worker_num)

    train_claim_cls(net_clc, loss_criterion, opti_clc, train_loader, dev_loader, dev_claims, gpu)

    net_clc.load_state_dict(torch.load(clc_model_params_filename))

    return net_clc


if __name__ == '__main__':
    pass