import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dataset_loader import load_data

def tfidf_cos_baseline(claims, evidences, evidence_select_num=10):
    recall, precision = 0.0, 0.0
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
        recall += len([e for e in evidences if e in e_true])/ len(e_true)
        precision += len([e for e in evidences if e in e_true]) / len(evidences)

    recall /= len(claim_evidences)
    precision /= len(claim_evidences)

    if recall + precision == 0.0:
        return 0.0, 0.0, 0.0
    else:
        return 2 * (precision * recall) / (precision + recall), recall, precision  # F1 Score, recall, precision


if __name__ == '__main__':
    train_claims, dev_claims, test_claims, evidences = load_data()

    f1, recall, precision = tfidf_cos_baseline(dev_claims, evidences)
    print("------Evidence Retrival Baseline Performance------")
    print(f"F1-Score: {f1}")
    print(f"Recall-Score: {recall}")
    print(f"Precision-Score: {precision}")
    print("--------------------------------------------------")