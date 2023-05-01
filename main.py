from bert_er import *
from bert_clc import *
from dataset_loader import load_data


def CFEVER_main():
    _, _, test_claims, evidences = load_data()

    net_er = er_pipeline()

    test_set = CFEVERERTestDataset(test_claims, evidences)
    test_loader = DataLoader(test_set, batch_size=loader_batch_size, num_workers=loader_worker_num)

    claim_evidences = predict_evi(net_er, test_loader, gpu)
    test_claims = extract_er_result(claim_evidences, test_claims)