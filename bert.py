import json
import math
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
from sklearn.metrics.pairwise import cosine_similarity
#import tensorflow_hub as hub


use_module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
evidence_key_prefix = 'evidence-'
d_bert_base = 768
d_bert_large = 1024
max_evi = 5
gpu = 0
evidence_selection_threshold = 0.5
input_seq_max_len = 384

class CFEVERERDataset(Dataset):
    """Climate Fact Extraction and Verification Dataset for Training, for the Evidence Retrival task."""

    def __init__(self, claims, evidences_, is_train, max_len=input_seq_max_len):
        self.data_set = unroll_claim_evidences(claims, evidences_, is_train, sample_ratio=1)
        self.max_len = max_len
        self.claims = claims
        self.evidences = evidences_
        self.is_train  = is_train

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, index):
        if self.is_train:
            claim_id, evidence_id, label = self.data_set[index]
        else:
            claim_id, evidence_id = self.data_set[index]

        # Preprocessing the text to be suitable for BERT
        claim_evidence_in_tokens = self.tokenizer.encode_plus(self.claims[claim_id]['claim_text'], self.evidences[evidence_id], 
                                                              return_tensors='pt', padding='max_length', truncation=True,
                                                              max_length=self.max_len, return_token_type_ids=True)
        
        seq, attn_masks, segment_ids = claim_evidence_in_tokens['input_ids'].squeeze(0), claim_evidence_in_tokens[
                'attention_mask'].squeeze(0), claim_evidence_in_tokens['token_type_ids'].squeeze(0)
    
        return (seq, attn_masks, segment_ids, label) if self.is_train else (seq, attn_masks, segment_ids, claim_id, evidence_id)


def generate_train_evidence_samples(full_evidences_len, claim_evidences, sample_ratio):
    """
    Generate training samples for each of the claims for the evidence retrieval task.
    :param full_evidences: the full evidence set.
    :param claim_evidences: the ground truth evidence set for the claim. In the form of a list of evidence ids
    :param sample_ratio: the ratio of positive to negative samples: pos/neg
    :return: a list of evidence samples.
    """

    # Get positive samples
    samples = claim_evidences  # evidence ids

    # Get negative samples
    samples += random.sample([evidence_key_prefix + str(i) for i in range(full_evidences_len)
                              if evidence_key_prefix + str(i) not in claim_evidences],
                             len(claim_evidences) * sample_ratio)  # random selection

    samples_with_labels = list(zip(samples, [1] * len(claim_evidences) + [0] * (len(samples) - len(claim_evidences))))

    return samples_with_labels


def generate_test_evidence_candidates(full_evidences, claim_text, max_candidates=1000):
    """
    Generate test samples for each of the claims for the evidence retrieval task.
    :param full_evidences: the full evidence set.
    :return: a list of evidence samples.
    """

    # Get negative samples
    samples = list(full_evidences.keys())  # np.array(list(full_evidences.items()))[:, 0]

    return samples


def unroll_claim_evidences(claims, evidences_, is_train, sample_ratio=1):
    st = time.time()
    if is_train:
        full_evidences_len = len(evidences_)
        train_claim_evidence_pairs = []
        for claim in claims:
            for train_evidence_id, label in generate_train_evidence_samples(full_evidences_len,
                                                    claims[claim]['evidences'], sample_ratio):
                train_claim_evidence_pairs.append((claim, train_evidence_id, label))

        random.shuffle(train_claim_evidence_pairs)
        print(f"Finished unrolling train claim-evidence pairs in {time.time() - st} seconds.")

        return train_claim_evidence_pairs
    else:
        # Load the Universal Sentence Encoder
        #use_model = hub.load(use_module_url)
        test_claim_evidence_pairs = []
        for claim in claims:
            for test_evidence_id in generate_test_evidence_candidates(evidences_, claims[claim]['claim_text']):
                test_claim_evidence_pairs.append((claim, test_evidence_id))

        print(f"Finished unrolling test claim-evidence pairs in {time.time() - st} seconds.")

        return test_claim_evidence_pairs


class CFEVERERClassifier(nn.Module):

    def __init__(self):
        super(CFEVERERClassifier, self).__init__()

        # Instantiating BERT model object
        self.bert = BertModel.from_pretrained('bert-base-uncased')

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

        # Feeding cls_rep to the classifier layer
        logits = self.cls_layer(cls_rep)

        return logits


def train_evi_retrival(net, loss_criterion, opti, train_loader, dev_loader, max_eps, dev_claims, gpu):
    best_acc = 0
    st = time.time()

    for ep in range(max_eps):
        net.train()  # Good practice to set the mode of the model

        for i, (seq, attn_masks, segment_ids, labels) in enumerate(train_loader):
            # Reset/Clear gradients
            opti.zero_grad()

            # Extracting the tokens ids, attention masks and token type ids
            seq, attn_masks, segment_ids, labels = seq.cuda(gpu), attn_masks.cuda(gpu), segment_ids.cuda(gpu), labels.cuda(gpu)

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
                print("Iteration {} of epoch {} complete. Loss: {}; Accuracy: {}; Time taken (s): {}".format(i, ep, loss.item(), acc, (time.time() - st)))
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


def get_probs_from_logits(logits, threshold=0.5):
    probs = torch.sigmoid(logits.unsqueeze(-1))

    return probs.squeeze()


def reselect_candidates(existing_candidates, new_candidate):
    return sorted(existing_candidates + [new_candidate], key=lambda x: x[0], reverse=True)[:max_evi]


def evaluate(net, loss_criterion, dataloader, dev_claims, gpu):
    net.eval()

    # claim_evidences = defaultdict(lambda: [(-math.inf, None)] * max_evi)
    # recall, precision = 0.0, 0.0

    # with torch.no_grad():  # suspend grad track, save time and memory
    #     for seq, attn_masks, segment_ids, claim_ids, evidence_ids in dataloader:
    #         seq, attn_masks, segment_ids = seq.cuda(gpu), attn_masks.cuda(gpu), segment_ids.cuda(gpu)
    #         logits = net(seq, attn_masks, segment_ids)
    #         probs = get_probs_from_logits(logits)
            
    #         for i, prob in enumerate(probs):
    #             claim_evidences[claim_ids[i]] = reselect_candidates(claim_evidences[claim_ids[i]], (prob.item(), evidence_ids[i]))  
    

    # for claim_id in claim_evidences.keys():
    #     claim_evidences[claim_id] = [evidence[1] for evidence in claim_evidences[claim_id] if evidence[0] > evidence_selection_threshold]

    # for claim_id, evidences in claim_evidences.items():
    #     e_true = dev_claims[claim_id]['evidences']
    #     recall += len(set(e_true).intersection(set(evidences))) / len(e_true)
    #     precision += len(set(e_true).intersection(set(evidences))) / len(evidences)

    # recall /= len(claim_evidences)
    # precision /= len(claim_evidences)

    # return 2 * (precision * recall) / (precision + recall)  # F1 Score

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
    biggest_claim = (0, "")
    biggest = (0, "", "")
    count = 0
    for claim in train_claims:
        biggest_claim = max([biggest_claim, (len(train_claims[claim]['claim_text'].split()), train_claims[claim]['claim_text'])],
                            key=lambda x: x[0])
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
    print("\n\n")
    print("Biggest claim: ", biggest_claim)


if __name__ == '__main__':
    random.seed(42)

    train_path = '../project-data/train-claims.json'
    dev_path = '../project-data/dev-claims.json'
    test_path = '../project-data/test-claims-unlabelled.json'
    evidence_path = '../project-data/evidence.json'

    train_claims, dev_claims, test_claims, evidences = load_data(train_path, dev_path, test_path, evidence_path)

    # Creating instances of training and development set
    # max_len sets the maximum length that a sentence can have,
    # any sentence longer than that length is truncated to the max_len size
    train_set = CFEVERERDataset(train_claims, evidences, True)
    # dev_set = CFEVERERDataset(train_claims, evidences, False)

    # Creating intsances of training and development dataloaders
    train_loader = DataLoader(train_set, batch_size=24, num_workers=2)
    # dev_loader = DataLoader(dev_set, batch_size=24, num_workers = 2)

    net = CFEVERERClassifier()
    net.cuda(gpu) #Enable gpu support for the model

    loss_criterion = nn.BCEWithLogitsLoss()
    opti = optim.Adam(net.parameters(), lr=2e-5)

    num_epoch = 1

    # fine-tune the model
    train_evi_retrival(net, loss_criterion, opti, train_loader, None, num_epoch, dev_claims, gpu)