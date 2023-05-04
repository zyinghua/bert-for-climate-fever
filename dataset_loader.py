# This file is used to load the datasets from the json files.

import json

train_path = '../project-data/train-claims.json'
dev_path = '../project-data/dev-claims.json'
test_path = '../project-data/test-claims-unlabelled.json'
evidence_path = '../project-data/evidence.json'

def load_data():
    train_claims = json.load(open(train_path))
    dev_claims = json.load(open(dev_path))
    test_claims = json.load(open(test_path))
    evidences = json.load(open(evidence_path))

    return train_claims, dev_claims, test_claims, evidences