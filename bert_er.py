# This file contains all the code for the Evidence Retrival task. Data -> Training -> ER Model

import json
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random
from transformers import BertTokenizer
from transformers import BertModel
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from transformers import AdamW
import time
import copy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from torch.optim.lr_scheduler import CosineAnnealingLR
from main import path_prefix

random.seed(42)
evidence_key_prefix = 'evidence-'
er_result_filename = path_prefix + "evidence-retrival-only-results.json"
er_model_params_filename = path_prefix + 'cfeverercls.dat'
claim_hard_negatives_filename = path_prefix + 'claim-hard-negative-evidences.json'

# ----------Hyperparameters of the entire pipeline----------
# --------------Evidence Retrival--------------
d_bert_base = 768
gpu = 0
input_seq_max_len = 384
data_aug_scale = 3
pre_select_evidence_num = 1000
loader_batch_size = 24
loader_worker_num = 2
num_epoch_pre = 5
num_epoch_post = 12
hnm_threshold = 0.7
hnm_batch_size = 12
evidence_selection_threshold = 0.9
max_evi = 5
opti_lr_er_pre_s1 = 1e-5
opti_lr_er_pre_s2 = 1e-6
opti_lr_er_hne = 2e-7
grad_step_period_pre = 1
grad_step_period_hne = 2
# ----------------------------------------------


class CFEVERERTrainDataset(Dataset):
    """Climate Fact Extraction and Verification Dataset for Train, for the Evidence Retrival task."""

    def __init__(self, claims, evidences_, tokenizer, max_len=input_seq_max_len, data_aug_scale=data_aug_scale):
        self.data_set = unroll_train_claim_evidences(claims, evidences_, data_aug_scale=data_aug_scale)
        self.max_len = max_len
        self.claims = claims
        self.evidences = evidences_
        self.data_aug_scale=data_aug_scale
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data_set)
    
    def reset_data_random(self):
        self.data_set = unroll_train_claim_evidences(self.claims, self.evidences, self.data_aug_scale)

    def reset_data_hne(self, claim_hard_negative_evidences):
        self.data_set = unroll_train_claim_evidences_with_hne(self.claims, self.evidences, claim_hard_negative_evidences)

    def __getitem__(self, index):
        claim_id, evidence_id, label = self.data_set[index]

        # Preprocessing the text to be suitable for BERT
        claim_evidence_in_tokens = self.tokenizer.encode_plus(self.claims[claim_id]['claim_text'], self.evidences[evidence_id], 
                                                              return_tensors='pt', padding='max_length', truncation=True,
                                                              max_length=self.max_len, return_token_type_ids=True)
        
        seq, attn_masks, segment_ids = claim_evidence_in_tokens['input_ids'].squeeze(0), claim_evidence_in_tokens[
                'attention_mask'].squeeze(0), claim_evidence_in_tokens['token_type_ids'].squeeze(0)
    
        return seq, attn_masks, segment_ids, label


class CFEVERERTestDataset(Dataset):
    """Climate Fact Extraction and Verification Dataset for Dev/Test, for the Evidence Retrival task."""

    def __init__(self, claims, evidences_, tokenizer, max_len=input_seq_max_len, max_candidates=pre_select_evidence_num):
        self.data_set = unroll_test_claim_evidences(claims, evidences_, max_candidates=max_candidates)
        self.max_len = max_len
        self.claims = claims
        self.evidences = evidences_
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, index):
        claim_id, evidence_id = self.data_set[index]

        # Preprocessing the text to be suitable for BERT
        claim_evidence_in_tokens = self.tokenizer.encode_plus(self.claims[claim_id]['claim_text'], self.evidences[evidence_id], 
                                                              return_tensors='pt', padding='max_length', truncation=True,
                                                              max_length=self.max_len, return_token_type_ids=True)
        
        seq, attn_masks, segment_ids = claim_evidence_in_tokens['input_ids'].squeeze(0), claim_evidence_in_tokens[
                'attention_mask'].squeeze(0), claim_evidence_in_tokens['token_type_ids'].squeeze(0)
    
        return seq, attn_masks, segment_ids, claim_id, evidence_id


def unroll_train_claim_evidences(claims, evidences_, data_aug_scale):
    """
    This function aims to define the train evidences for each claim, 
    unroll them into pairs, and return a list of claim-evidence pairs
    in the form of (claim_id, evidence_id, label).

    Rule: Includes all the positive evidences for each claim, and randomly
    sample negative evidences for each claim, number of negative evidences
    is determined by the sample_ratio.
    """
    st = time.time()

    train_claim_evidence_pairs = []

    for i in range(data_aug_scale):
        for claim in claims:
            for train_evidence_id, label in generate_train_evidence_samples(evidences_, claims[claim]['evidences']):
                train_claim_evidence_pairs.append((claim, train_evidence_id, label))

    random.shuffle(train_claim_evidence_pairs)
    print(f"Finished unrolling train claim-evidence pairs in {time.time() - st} seconds.")

    return train_claim_evidence_pairs


def unroll_test_claim_evidences(claims, evidences_, max_candidates):
    """
    This function aims to define the evidences to be further processed
    by the BERT model for each test claim. The evidences are unrolled
    into pairs, and return a list of claim-evidence pairs in the form
    of (claim_id, evidence_id).

    Rule: Includes the top <max_candidates> evidences for each claim 
    based on the TF-IDF cosine similarity score with the corresponding
    claim.
    """
    st = time.time()

    vectorizer = TfidfVectorizer(stop_words='english')
    vectorizer.fit(list(evidences_.values()) + [claims[c]["claim_text"] for c in claims])
    evidences_tfidf = vectorizer.transform(evidences_.values())

    test_claim_evidence_pairs = []
    for claim in claims:
        claim_tfidf = vectorizer.transform([claims[claim]['claim_text']])

        for test_evidence_id in generate_test_evidence_candidates(evidences_, evidences_tfidf, claim_tfidf, max_candidates):
            test_claim_evidence_pairs.append((claim, test_evidence_id))

    print(f"Finished unrolling test claim-evidence pairs in {time.time() - st} seconds.")

    return test_claim_evidence_pairs


def generate_train_evidence_samples(evidences_, claim_evidences, sample_ratio=1):
    """
    Generate training samples for each of the claims for the evidence retrieval task.
    :param evidences_: the full evidence set.
    :param claim_evidences: the ground truth evidence set for the claim. In the form of a list of evidence ids
    :param sample_ratio: the ratio of positive to negative samples: neg/pos
    :return: a list of evidence samples zipped with the corresponding labels. - (evi id, label)
    """
        
    # Get positive samples
    samples = claim_evidences.copy()  # evidence ids

    # Get negative samples
    while len(samples) < math.ceil(len(claim_evidences) * (sample_ratio + 1)):
        neg_sample = evidence_key_prefix + str(random.randint(0, len(evidences_) - 1))  # random selection
        
        if neg_sample not in samples:
            samples.append(neg_sample)

    samples_with_labels = list(zip(samples, [1] * len(claim_evidences) + [0] * (len(samples) - len(claim_evidences))))

    return samples_with_labels


def generate_test_evidence_candidates(evidences_, evidences_tfidf, claim_tfidf, max_candidates):
    """
    :param evidences_: the full evidence set.
    :param evidences_tfidf: The tfidf matrix of the entire evidence set
    :param claim_tfidf: The tfidf vector of the query claim (also a matrix technically).
    :param max_candidates: Number of evidences to be selected for further processing.
    :return: a list of the selected evidences.
    """
    similarity = cosine_similarity(claim_tfidf, evidences_tfidf).squeeze()
    
    df = pd.DataFrame({"evidences": evidences_.keys(), "similarity": similarity}).sort_values(by=['similarity'], ascending=False)
    potential_relevant_evidences = df.iloc[:max_candidates]["evidences"].tolist()

    return potential_relevant_evidences


class CFEVERERClassifier(nn.Module):
    def __init__(self, dropout_prob=0.1):
        super(CFEVERERClassifier, self).__init__()

        # Instantiating BERT model object
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        #self.dropout = nn.Dropout(dropout_prob)

        # Classification layer
        # input dimension is 768 because [CLS] embedding has a dimension of 768, if bert base is used
        # output dimension is 1 because we're working with a binary classification problem - RELEVANT : NOT RELEVANT
        self.cls_layer = nn.Linear(d_bert_base, 1)

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

        # Apply dropout
        #cls_rep = self.dropout(cls_rep)

        # Feeding cls_rep to the classifier layer
        logits = self.cls_layer(cls_rep)

        return logits


def train_evi_retrival(net, loss_criterion, opti, train_loader, dev_loader, train_set, dev_claims, gpu, max_eps, grad_step_period, claim_hard_negative_evidences=None):
    best_f1 = 0
    mean_losses = []

    if claim_hard_negative_evidences is None:
        scheduler = CosineAnnealingLR(opti, T_max=max_eps, eta_min=opti_lr_er_pre_s2)
    
    for ep in range(max_eps):
        net.train()  # Good practice to set the mode of the model
        st = time.time()
        opti.zero_grad()
        count = 0
        
        for i, (seq, attn_masks, segment_ids, labels) in enumerate(train_loader):
            # Extracting the tokens ids, attention masks and token type ids
            seq, attn_masks, segment_ids, labels = seq.cuda(gpu), attn_masks.cuda(gpu), segment_ids.cuda(gpu), labels.cuda(gpu)

            # Obtaining the logits from the model
            logits = net(seq, attn_masks, segment_ids)

            # Computing loss
            loss = loss_criterion(logits.squeeze(-1), labels.float())
            mean_losses[ep] += loss.item()
            count += 1

            # Backpropagating the gradients, account for gradients
            loss.backward()

            if (i + 1) % grad_step_period == 0:
                # Optimization step, apply the gradients
                opti.step()

                # Reset/Clear gradients
                opti.zero_grad()

            if i % 100 == 0:
                acc = get_accuracy_from_logits(logits, labels)
                print("Iteration {} of epoch {} complete. Loss: {}; Accuracy: {}; Time taken (s): {}".format(i, ep, loss.item(), acc, (time.time() - st)))
                st = time.time()
        
        mean_losses[ep] /= count

        if claim_hard_negative_evidences is None:
            scheduler.step()
        
        # if claim_hard_negative_evidences is None and ep == 2:  # slow down learning rate
        #     opti.param_groups[0]["lr"] = opti_lr_er_pre_s2
        
        if (ep + 1) % 1 == 0:
            dev_st = time.time()
            print("Evaluating on the dev set... (This might take a while)")
            f1, recall, precision = evaluate(net, dev_loader, dev_claims, gpu)
            print("\nEpoch {} completed! Evaluation on dev set took {} seconds.\nDevelopment F1: {}; Development Recall: {}; Development Precision: {}".format(ep, time.time() - dev_st, f1, recall, precision))
            
            if f1 > best_f1:
                print("Best development f1 improved from {} to {}, saving model...\n".format(best_f1, f1))
                best_f1 = f1
                torch.save(net.state_dict(), er_model_params_filename)
            else:
                print()
    
    return mean_losses


def get_accuracy_from_logits(logits, labels):
    probs = torch.sigmoid(logits.unsqueeze(-1))
    preds = (probs > 0.5).long()
    acc = (preds.squeeze() == labels).float().mean()
    return acc


def get_probs_from_logits(logits):
    probs = torch.sigmoid(logits.unsqueeze(-1))

    return probs.squeeze()


def select_evi_df(df, threshold, max_evidences):
    """
    Selects the top <max_evidences> evidences from the 
    dataframe <df> with a probability higher than <threshold>.
    If no one satisifies the threshold, the evidence with the highest
    probability is selected.
    """
    
    max_prob_evi = df[df['probs'] == df['probs'].max()]

    df = df[df['probs'] > threshold].nlargest(max_evidences, "probs")

    if len(df) == 0:
        df = max_prob_evi

    return df

def predict_evi(net, dataloader, gpu, threshold=evidence_selection_threshold, max_evidences=max_evi, evaluate=False, evaluation_claims=None, loss_criterion=None):
    net.eval()

    claim_evidences = defaultdict(list)
    df = pd.DataFrame()
    mean_loss = 0

    with torch.no_grad():  # suspend grad track, save time and memory
        for seq, attn_masks, segment_ids, claim_ids, evidence_ids in dataloader:
            if evaluate and evaluation_claims is not None:
                labels = torch.tensor([1 if evidence_ids[i] in evaluation_claims[claim_ids[i]]['evidences'] else 0 for i in range(len(claim_ids))])
                labels = labels.cuda(gpu)

            seq, attn_masks, segment_ids = seq.cuda(gpu), attn_masks.cuda(gpu), segment_ids.cuda(gpu)
            logits = net(seq, attn_masks, segment_ids)
            probs = get_probs_from_logits(logits)

            if evaluate:
                mean_loss += loss_criterion(logits.squeeze(-1), labels.float()).item()
            
            df = pd.concat([df, pd.DataFrame({"claim_ids": claim_ids, "evidence_ids": evidence_ids, "probs": probs.cpu()})], ignore_index=True)

    # groupby gives a df for each claim_ids, then for each df, apply() the selection, finally reset_index to get rid of the multi-index
    filtered_claim_evidences_df = df.groupby('claim_ids').apply(lambda x: select_evi_df(x, threshold, max_evidences)).reset_index(drop=True)

    for _, row in filtered_claim_evidences_df.iterrows():
        claim_id = row['claim_ids']
        evidence_id = row['evidence_ids']

        claim_evidences[claim_id].append(evidence_id)
    
    return claim_evidences if not evaluate else (claim_evidences, mean_loss / len(dataloader))


def evaluate(net, dataloader, dev_claims, loss_criterion, gpu):
    """
    Used to evaluate the dev set performance of the model.
    """
    claim_evidences, loss = predict_evi(net, dataloader, gpu, evaluate=True, evaluation_claims=dev_claims, loss_criterion=loss_criterion)

    fscores, recalls, precisions = [], [], []

    for claim_id, evidences in claim_evidences.items():
        e_true = dev_claims[claim_id]['evidences']
        recall = len([e for e in evidences if e in e_true]) / len(e_true)
        precision = len([e for e in evidences if e in e_true]) / len(evidences)
        fscore = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0.0

        fscores.append(fscore)
        precisions.append(precision)
        recalls.append(recall)

    mean_f = np.mean(fscores if len(fscores) > 0 else [0.0])
    mean_recall = np.mean(recalls if len(recalls) > 0 else [0.0])
    mean_precision = np.mean(precisions if len(precisions) > 0 else [0.0])

    return mean_f, mean_recall, mean_precision, loss  # F1 Score, recall, precision, loss


def extract_er_result(claim_evidences, claims, filename=er_result_filename):
    """
    Extract the evidences from the claim_evidences dict and
    save the result to a json file. This step only considers
    the evidences for a claim, with no care to the labels.
    """
    extracted_claims = copy.deepcopy(claims)

    for c in extracted_claims:
        extracted_claims[c]["evidences"] = claim_evidences[c]
    
    with open(filename, 'w') as f:
        json.dump(extracted_claims, f)

    return extracted_claims


class CFEVERERHNMDataset(Dataset):
    """
    This dataset is used to obtain the hard negative evidences for a given claim
    for a pre-trained ER model. All evidences that are not positive for the claim
    are considered in the dataset.

    Note: This dataset only takes one claim instead of all like in the normal train
    dataset above. Because hard negative evidences are selected for a claim at a time.
    """
    def __init__(self, claim, evidences_, tokenizer, max_len=input_seq_max_len):
        self.data_set = [e for e in evidences_ if e not in claim['evidences']]  # get all negative samples
        self.max_len = max_len
        self.claim = claim
        self.evidences = evidences_
        self.target_hn_num = len(claim['evidences'])  # number of hard negative evidences to be selected
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, index):
        evidence_id = self.data_set[index]

        # Preprocessing the text to be suitable for BERT
        claim_evidence_in_tokens = self.tokenizer.encode_plus(self.claim['claim_text'], self.evidences[evidence_id], 
                                                              return_tensors='pt', padding='max_length', truncation=True,
                                                              max_length=self.max_len, return_token_type_ids=True)
        
        seq, attn_masks, segment_ids = claim_evidence_in_tokens['input_ids'].squeeze(0), claim_evidence_in_tokens[
                'attention_mask'].squeeze(0), claim_evidence_in_tokens['token_type_ids'].squeeze(0)
    
        return seq, attn_masks, segment_ids, evidence_id


def hnm(net, train_claims, evidences_, gpu, hnm_threshold=hnm_threshold, hnm_batch_size=hnm_batch_size):
    """
    This function aims to select the hard negative evidences for each claim.
    returns a dict of claim_id -> list of hard negative evidences.
    """
    net.eval()
    st = time.time()

    claim_hard_negative_evidences = defaultdict(list)  # store the hard negative evidences for each claim
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    for k, train_claim in enumerate(train_claims):  # for each claim in the training set
        test_train_set = CFEVERERHNMDataset(train_claims[train_claim], evidences_, tokenizer)  # get the dataset containing the negative evi for the claim
        test_train_loader = DataLoader(test_train_set, batch_size=hnm_batch_size, num_workers=loader_worker_num)

        with torch.no_grad():  # suspend grad track, save time and memory
            for seq, attn_masks, segment_ids, evidence_ids in test_train_loader:  
                seq, attn_masks, segment_ids = seq.cuda(gpu), attn_masks.cuda(gpu), segment_ids.cuda(gpu)
                logits = net(seq, attn_masks, segment_ids)
                probs = get_probs_from_logits(logits)

                indices = np.where(probs.cpu().numpy() > hnm_threshold)[0]  # get the indices of the hard negative evidences if any
                i = 0

                while len(claim_hard_negative_evidences[train_claim]) < test_train_set.target_hn_num and i < len(indices):
                    """While the number of hard negative evidences for the claim is less than the target number,
                    and there are still hard negative evidences in the indices, add the evidences to the list."""
                    claim_hard_negative_evidences[train_claim].append(evidence_ids[indices[i]])
                    i += 1

                if len(claim_hard_negative_evidences[train_claim]) == test_train_set.target_hn_num:  # if the enough hard negatives, break
                    break
        
        if k % 50 == 0:
            print(f"{k}th claim finished in {time.time() - st} seconds.")
            st = time.time()
    
    with open(claim_hard_negatives_filename, 'w') as f:
        json.dump(claim_hard_negative_evidences, f)
        print("\nClaim hard negative evidences saved to file.")

    return claim_hard_negative_evidences


def unroll_train_claim_evidences_with_hne(claims, evidences_, claim_hard_negative_evidences, hne_sample_ratio=0.5):
    st = time.time()

    train_claim_evidence_pairs = []

    for claim in claims:
        for train_evidence_id, label in generate_train_evidence_samples(evidences_, claims[claim]['evidences'], hne_sample_ratio):
            train_claim_evidence_pairs.append((claim, train_evidence_id, label))

        for train_evidence_id in claim_hard_negative_evidences[claim]:
            train_claim_evidence_pairs.append((claim, train_evidence_id, 0))

    random.shuffle(train_claim_evidence_pairs)
    print(f"Finished unrolling train claim-evidence pairs with hne in {time.time() - st} seconds.")

    return train_claim_evidence_pairs


def er_pipeline(train_claims, dev_claims, evidences):
    #-------------------------------------------------------------
    net_er = CFEVERERClassifier()
    net_er.cuda(gpu) # Enable gpu support for the model

    loss_criterion = nn.BCEWithLogitsLoss()
    opti_er_pre = optim.Adam(net_er.parameters(), lr=opti_lr_er_pre)
    opti_er_hne = AdamW(net_er.parameters(), lr=opti_lr_er_hne, weight_decay=0.15)
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Creating instances of training, test and development set
    train_set = CFEVERERTrainDataset(train_claims, evidences, bert_tokenizer)
    dev_set = CFEVERERTestDataset(dev_claims, evidences, bert_tokenizer)

    #Creating intsances of training, test and development dataloaders
    train_loader = DataLoader(train_set, batch_size=loader_batch_size, num_workers=loader_worker_num)
    dev_loader = DataLoader(dev_set, batch_size=loader_batch_size, num_workers=loader_worker_num)

    # First phrase: pre-train the model on all positive claim-evidence pairs and same number of random negative pairs
    train_evi_retrival(net_er, loss_criterion, opti_er_pre, train_loader, dev_loader, train_set, dev_claims, gpu, num_epoch_pre, grad_step_period_pre)

    net_er.load_state_dict(torch.load(er_model_params_filename))  # load the best model
    claim_hard_negative_evidences = hnm(net_er, train_claims, evidences, gpu)
    # claim_hard_negative_evidences = json.load(open(claim_hard_negatives_filename, 'r'))

    train_evi_retrival(net_er, loss_criterion, opti_er_hne, train_loader, dev_loader, train_set, dev_claims, gpu, num_epoch_post, grad_step_period_hne, claim_hard_negative_evidences=claim_hard_negative_evidences)

    net_er.load_state_dict(torch.load(er_model_params_filename))
    return net_er
    #-------------------------------------------------------------


if __name__ == '__main__':
    pass
    # 2e-7 with ep = 13, F1s: [0.20268501339929915, 0.2002319109461967, 0.19967532467532473, 0.20389610389610394, 
    # 0.20436507936507942, 0.20454545454545459, 0.20310245310245312, 0.20625644197072773， 0.2100082457225315，0.2116316223459081， 0.2177076891362606， 0.21554318697175848, 0.2126211090496805]

    # 