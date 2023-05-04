# This file contains the code of a baseline model for the evidence retrival task.

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dataset_loader import load_data

def tfidf_cos_er_baseline(claims, evidences, evidence_select_num=10):
    """
    Selects the K most cosine similar evidences based on TF-IDF.
    """
    fscores, recalls, precisions = [], [], []
    claim_evidences = {}

    vectorizer = TfidfVectorizer(stop_words='english')
    vectorizer.fit(list(evidences.values()) + [claims[c]["claim_text"] for c in claims])
    evidences_tfidf = vectorizer.transform(evidences.values())

    for c in claims:
        claim_tfidf = vectorizer.transform([claims[c]["claim_text"]])

        cos_sims = cosine_similarity(claim_tfidf, evidences_tfidf).squeeze()
    
        df = pd.DataFrame({"evidences": evidences.keys(), "similarity": cos_sims}).sort_values(by=['similarity'], ascending=False)
        claim_evidences[c] = df.iloc[:evidence_select_num]["evidences"].tolist()
    
    for claim_id, evidences in claim_evidences.items():
        e_true = claims[claim_id]['evidences']
        recall = len([e for e in evidences if e in e_true]) / len(e_true)
        precision = len([e for e in evidences if e in e_true]) / len(evidences)
        fscore = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0.0

        fscores.append(fscore)
        precisions.append(precision)
        recalls.append(recall)

    mean_f = np.mean(fscores if len(fscores) > 0 else [0.0])
    mean_recall = np.mean(recalls if len(recalls) > 0 else [0.0])
    mean_precision = np.mean(precisions if len(precisions) > 0 else [0.0])

    return mean_f, mean_recall, mean_precision  # F1 Score, recall, precision


def run_er_baseline(dev_claims, evidences):
    f1, recall, precision = tfidf_cos_er_baseline(dev_claims, evidences)
    print("------Evidence Retrival Baseline Performance------")
    print(f"F1-Score: {f1}")
    print(f"Recall-Score: {recall}")
    print(f"Precision-Score: {precision}")
    print("--------------------------------------------------")


def zero_r_label_cls_baseline(train_claims, dev_claims):
    acc = 0
    majority_label = Counter([train_claims[c]["claim_label"] for c in train_claims]).most_common(1)[0][0]

    for c in dev_claims:
        if dev_claims[c]['claim_label'] == majority_label:
            acc += 1
    
    return acc / len(dev_claims)


if __name__ == '__main__':
    train_claims, dev_claims, test_claims, evidences = load_data()

    print(zero_r_label_cls_baseline(train_claims, dev_claims))