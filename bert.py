import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random
from transformers import BertTokenizer
from transformers import BertModel
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import time
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow_hub as hub


use_module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
evidence_key_prefix = 'evidence-'
d_bert_base = 768
d_bert_large = 1024
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
gpu = 0


class CFEVERERTrainDataset(Dataset):
    """Climate Fact Extraction and Verification Dataset for Training, for the Evidence Retrival task."""

    def __init__(self, claims, evidences_, is_train, max_len=512):
        self.train_set = unroll_claim_evidences(claims, evidences_, is_train, sample_ratio=1)
        self.max_len = max_len

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.train_set)

    def __getitem__(self, index):
        train_claim_text, train_evidence, label = self.train_set[index]

        # Preprocessing the text to be suitable for BERT
        claim_evidence_in_tokens = self.tokenizer.encode_plus(train_claim_text, train_evidence, return_tensors='pt',
                                                              padding='max_length', truncation=True,
                                                              max_length=self.max_len, return_token_type_ids=True)
        
        seq, attn_masks, segment_ids = claim_evidence_in_tokens['input_ids'], claim_evidence_in_tokens[
                'attention_mask'], claim_evidence_in_tokens['token_type_ids']
    
        return seq, attn_masks, segment_ids, label


class CFEVERERTestDataset(Dataset):
    """Climate Fact Extraction and Verification Dataset for Testing, for the Evidence Retrival task."""

    def __init__(self, claims, evidence_, max_len=512):

        self.evidence_candidates = evidence_  # In the form of (evidence_id, evidence_text)
        self.max_len = max_len 
        self.claims = claims

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        claim_text = self.df.loc[self.df.index[index], 'claim_text']
        claim_evidences = self.df.loc[self.df.index[index], 'evidences']

        labels = [1 if id in claim_evidences else 0 for id, test_evidence in self.evidence_candidates]

        # Preprocessing the text to be suitable for BERT
        claim_evidences_in_tokens = [self.tokenizer.encode_plus(claim_text, test_evidence, return_tensors='pt',
                                                                padding='max_length', truncation=True,
                                                                max_length=self.max_len, return_token_type_ids=True)
                                     for id, test_evidence in self.evidence_candidates]

        return claim_evidences_in_tokens, labels


def generate_train_evidence_samples(full_evidences, claim_evidences, sample_ratio):
    """
    Generate training samples for each of the claims for the evidence retrieval task.
    :param full_evidences: the full evidence set.
    :param claim_evidences: the ground truth evidence set for the claim. In the form of a list of evidence ids
    :param sample_ratio: the ratio of positive to negative samples: pos/neg
    :return: a list of evidence samples.
    """

    # Get positive samples
    samples = []
    for claim_evidence in claim_evidences:
        samples.append(full_evidences[claim_evidence])

    # Get negative samples
    samples += random.sample([full_evidences[evidence_key_prefix + str(i)] for i in range(len(full_evidences))
                              if evidence_key_prefix + str(i) not in claim_evidences],
                             len(claim_evidences) * sample_ratio)  # random selection

    samples_with_labels = list(zip(samples, [1 if i < len(claim_evidences) else 0 for i in range(len(samples))]))

    return samples_with_labels


def generate_dev_evidence_samples(full_evidences, claim_evidences):
    samples = list(full_evidences.items())

    return [(s[1], 1 if s[0] in claim_evidences else 0) for s in samples]


def generate_test_evidence_samples(full_evidences, max_candidates=500):
    """
    Generate test samples for each of the claims for the evidence retrieval task.
    :param full_evidences: the full evidence set.
    :return: a list of evidence samples.
    """

    # Get negative samples
    samples = full_evidences

    return samples


def unroll_claim_evidences(claims, evidences_, is_train, sample_ratio=1):
    """Reduce size of evidences for computational efficiency."""
    """------------------------------------------------------"""
    # pos_neg_pool_ratio = 1
    # par_evidences = []

    # for claim in train_claims:
    #     par_evidences += train_claims[claim]['evidences']

    # unum_pos = len(set(par_evidences))
    # par_evidences += [evidence_key_prefix + str(i) for i in random.sample(range(len(evidences_)), unum_pos * pos_neg_pool_ratio)]
    # evidences_ = {e: evidences_[e] for e in set(par_evidences)}
    """------------------------------------------------------"""
    st = time.time()
    if is_train:
        train_claim_evidence_pairs = []
        for claim in claims:
            for train_evidence, label in generate_train_evidence_samples(evidences_, claims[claim]['evidences'],
                                                                        sample_ratio):
                train_claim_evidence_pairs.append((claims[claim]['claim_text'], train_evidence, label))

        random.shuffle(train_claim_evidence_pairs)
        print(f"Finished unrolling train claim-evidence pairs in {time.time() - st} seconds.")

        return train_claim_evidence_pairs
    else:
        test_claim_evidence_pairs = []
        for claim in claims:
            for train_evidence, label in generate_dev_evidence_samples(evidences_, claims[claim]['evidences']):
                test_claim_evidence_pairs.append((claims[claim]['claim_text'], train_evidence, label))

        print(f"Finished unrolling test claim-evidence pairs in {time.time() - st} seconds.")

        return test_claim_evidence_pairs


class CFEVERERClassifier(nn.Module):

    def __init__(self):
        super(CFEVERERClassifier, self).__init__()

        # Instantiating BERT model object
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Classification layer
        # input dimension is 768 because [CLS] embedding has a dimension of 768
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

        # Feeding cls_rep to the classifier layer
        logits = self.cls_layer(cls_rep)

        return logits


def train_evi_retrival(net, loss_criterion, opti, train_loader, dev_loader, max_eps, gpu):
    best_acc = 0
    st = time.time()

    for ep in range(max_eps):
        net.train()  # Good practice to set the mode of the model
        for i, (seq, attn_masks, segment_ids, labels) in enumerate(train_loader):
            # Reset/Clear gradients
            opti.zero_grad()

            # Extracting the tokens ids, attention masks and token type ids
            seq, attn_masks, segment_ids, labels = seq.to(device), attn_masks.to(device), segment_ids.to(device), labels.to(device)

            # Obtaining the logits from the model
            logits = net(seq, attn_masks, segment_ids)

            # Computing loss
            loss = loss_criterion(logits.squeeze(-1), labels.float())

            # Backpropagating the gradients, account for gradients
            loss.backward()

            # Optimization step, apply the gradients
            opti.step()

        if i % 100 == 0:
            acc = get_accuracy_from_logits(logits, labels)
            print("Iteration {} of epoch {} complete. Loss: {}; Accuracy: {}; Time taken (s): {}".format(i, ep,loss.item(), acc, (time.time() - st)))
            st = time.time()

    dev_acc, dev_loss = evaluate(net, loss_criterion, dev_loader, gpu)
    print("Epoch {} complete! Development Accuracy: {}; Development Loss: {}".format(ep, dev_acc, dev_loss))
    if dev_acc > best_acc:
        print("Best development accuracy improved from {} to {}, saving model...".format(best_acc, dev_acc))
        best_acc = dev_acc
        torch.save(net.state_dict(), 'cfeverercls_{}.dat'.format(ep))


def get_accuracy_from_logits(logits, labels):
    probs = torch.sigmoid(logits.unsqueeze(-1))
    preds = (probs > 0.5).long()
    acc = (preds.squeeze() == labels).float().mean()
    return acc


def evaluate(net, loss_criterion, dataloader, gpu):
    net.eval()

    mean_acc, mean_loss = 0, 0
    count = 0

    with torch.no_grad():  # suspend grad track, save time and memory
        for seq, attn_masks, labels in dataloader:
            seq, attn_masks, labels = seq.cuda(gpu), attn_masks.cuda(gpu), labels.cuda(gpu)
            logits = net(seq, attn_masks)
            mean_loss += loss_criterion(logits.squeeze(-1), labels.float()).item()
            mean_acc += get_accuracy_from_logits(logits, labels)
            count += 1

    return mean_acc / count, mean_loss / count


def load_data(train_path, dev_path, test_path, evidence_path):
    train_claims = json.load(open(train_path))
    dev_claims = json.load(open(dev_path))
    test_claims = json.load(open(test_path))
    evidences = json.load(open(evidence_path))

    return train_claims, dev_claims, test_claims, evidences


def get_use_embedding(model, sentence):
    return model([sentence])[0]


def get_similarity_score(claim, evidence, model):
    sen1_emb = get_use_embedding(model, claim)
    sen2_emb = get_use_embedding(model, evidence)

    # Compute the cosine similarity between the vectors
    cos_sim = cosine_similarity(np.array(list(sen1_emb)).reshape(1, -1), np.array(list(sen2_emb)).reshape(1, -1))[0][0]

    return cos_sim


def test_h():
    # print([evidences[e] for e in train_claims['claim-169']['evidences']])
    dict(train_claims).update(dev_claims)

    label_counter = Counter()
    evidence_num_counter = Counter()
    evidence_len_counter = Counter()
    biggest = (0, "", "")
    count = 0
    for claim in train_claims:
        label_counter.update([train_claims[claim]['claim_label']])
        evidence_num_counter.update([len(train_claims[claim]['evidences'])])
        for e in train_claims[claim]['evidences']:
            biggest = max([biggest, (len(evidences[e].split()), evidences[e], train_claims[claim]['claim_text'])],
                          key=lambda x: x[0])

        for i, e in enumerate(train_claims[claim]['evidences']):
            evidence_len_counter.update([len(evidences[e].split())])

        if train_claims[claim]['claim_label'] == 'REFUTES':
            print("------------------------------")
            print(f"Claim Label: {train_claims[claim]['claim_label']}")
            print(f"Claim text: {train_claims[claim]['claim_text']}")
            for i, e in enumerate(train_claims[claim]['evidences']):
                print(i, evidences[e])
            print("------------------------------")
            print("\n\n")

    print(
        f'Label Counter: {label_counter}, Len Counter:{evidence_num_counter}, Count: {count}, total: {len(train_claims)}, percentage: {count / len(train_claims)}')
    print("\n\n")
    print("Biggest evidence: ", biggest)


if __name__ == '__main__':
    random.seed(42)

    train_path = './project-data/train-claims.json'
    dev_path = './project-data/dev-claims.json'
    test_path = './project-data/test-claims-unlabelled.json'
    evidence_path = './project-data/evidence.json'

    train_claims, dev_claims, test_claims, evidences = load_data(train_path, dev_path, test_path, evidence_path)

    # Load the Universal Sentence Encoder

    use_model = hub.load(use_module_url)

    test_h()

    # # Creating instances of training and development set
    # # max_len sets the maximum length that a sentence can have,
    # # any sentence longer than that length is truncated to the max_len size
    # train_set = CFEVERERTrainDataset(train_claims, evidences, True)
    # # dev_set = CFEVERERTestDataset(train_claims, evidences, has_labels=True)

    # # #Creating intsances of training and development dataloaders
    # train_loader = DataLoader(train_set, batch_size=64, num_workers=4)
    # # dev_loader = DataLoader(dev_set, batch_size = 64, num_workers = 4)

    # net = CFEVERERClassifier()
    # # net.cuda(gpu) #Enable gpu support for the model

    # loss_criterion = nn.BCEWithLogitsLoss()
    # opti = optim.Adam(net.parameters(), lr=2e-5)

    # num_epoch = 1

    # # fine-tune the model
    # train_evi_retrival(net, loss_criterion, opti, train_loader, None, num_epoch, gpu)