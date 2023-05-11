# This file contains the code for the Claim Label Classification task: Data -> Training -> Classification Model

import json
import pandas as pd
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
input_seq_max_len = 512
loader_batch_size = 24
loader_worker_num = 2
num_epoch = 10
max_evi_num = 5
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
        evidences_combined = " ".join([self.evidences[eid] for eid in claim['evidences']])

        # Preprocessing the text to be suitable for BERT
        claim_evidence_in_tokens = self.tokenizer.encode_plus(claim['claim_text'], evidences_combined, 
                                                            return_tensors='pt', padding='max_length', truncation=True,
                                                            max_length=self.max_len, return_token_type_ids=True)
        
        seq, attn_masks, segment_ids = claim_evidence_in_tokens['input_ids'].squeeze(0), claim_evidence_in_tokens[
                'attention_mask'].squeeze(0), claim_evidence_in_tokens['token_type_ids'].squeeze(0)

        return seq, attn_masks, segment_ids, label_mapper_ltoi[claim['claim_label']]


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

        evidences_combined = " ".join([self.evidences[eid] for eid in claim['evidences']])

        # Preprocessing the text to be suitable for BERT
        claim_evidence_in_tokens = self.tokenizer.encode_plus(claim['claim_text'], evidences_combined, 
                                                            return_tensors='pt', padding='max_length', truncation=True,
                                                            max_length=self.max_len, return_token_type_ids=True)
        
        seq, attn_masks, segment_ids = claim_evidence_in_tokens['input_ids'].squeeze(0), claim_evidence_in_tokens[
                'attention_mask'].squeeze(0), claim_evidence_in_tokens['token_type_ids'].squeeze(0)

        return seq, attn_masks, segment_ids, claim_id


class CFEVERLabelClassifier(nn.Module):
    def __init__(self):
        super(CFEVERLabelClassifier, self).__init__()

        # Instantiating BERT model object
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Classification layer
        # input dimension is 768 because [CLS] embedding has a dimension of 768, if bert base is used
        # output dimension is 1 because we're working with a binary classification problem - RELEVANT : NOT RELEVANT
        self.cls_layer = nn.Linear(d_bert_base, num_of_classes)

    def forward(self, seq, attn_masks, segment_ids):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
            -segment_ids : Tensor of shape [B, T] containing token ids of segment embeddings (see BERT paper for more details)
        '''
        
        # Feeding the input to BERT model to obtain contextualized representations
        outputs = self.bert(seq, attention_mask=attn_masks, token_type_ids=segment_ids, return_dict=True)
        cont_reps = outputs.last_hidden_state

        # Obtaining the representation of [CLS] head (the first token)
        cls_rep = cont_reps[:, 0]

        # Feeding cls_rep to the classifier layer
        logits = self.cls_layer(cls_rep)

        return logits  # logits shape is [B, num_of_classes]


def train_claim_cls(net, loss_criterion, opti, train_loader, dev_loader, dev_claims, gpu, max_eps=num_epoch):
    best_acc = 0
    mean_losses = [0] * max_eps

    for ep in range(max_eps):
        net.train()  # Good practice to set the mode of the model
        st = time.time()
        train_acc = 0
        count = 0
        
        for i, (b_seq, b_attn_masks, b_segment_ids, b_label) in enumerate(train_loader):
            # Reset/Clear gradients
            opti.zero_grad()

            # Extracting the tokens ids, attention masks and token type ids
            b_seq, b_attn_masks, b_segment_ids, b_label = b_seq.cuda(gpu), b_attn_masks.cuda(gpu), b_segment_ids.cuda(gpu), b_label.cuda(gpu)

            # Obtaining the logits from the model
            logits = net(b_seq, b_attn_masks, b_segment_ids)

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
        print(f"Epoch {ep} completed. Loss: {mean_losses[ep]}, Accuracy: {train_acc / count}.")

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
        for b_seq, b_attn_masks, b_segment_ids, b_claim_id in dataloader:
            b_seq, b_attn_masks, b_segment_ids, = b_seq.cuda(gpu), b_attn_masks.cuda(gpu), b_segment_ids.cuda(gpu)
            logits = net(b_seq, b_attn_masks, b_segment_ids)

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