# This file contains the code to run the pipeline of the Climate FEVER system.

import bert_er as ber
import bert_clc as bclc
from dataset_loader import load_data
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import transformers

# !pip install torch torchvision transformers
# transformers.logging.set_verbosity_error()

path_prefix = '/content/drive/MyDrive/Colab Notebooks/Assignment3/'

# ----------------- Prediction -----------------
loader_batch_size = 24
loader_worker_num = 2
gpu = 0
# ----------------------------------------------

output_filename = path_prefix + 'test-claims-predictions.json'

def CFEVER_main():
    train_claims, dev_claims, test_claims, evidences = load_data()
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    net_er = ber.er_pipeline(train_claims, dev_claims, evidences)

    test_set_er = ber.CFEVERERTestDataset(test_claims, evidences, bert_tokenizer)
    test_loader_er = DataLoader(test_set_er, batch_size=loader_batch_size, num_workers=loader_worker_num)

    test_claims = ber.extract_er_result(ber.predict_evi(net_er, test_loader_er, gpu), test_claims)

    test_set_clc = bclc.CFEVERLabelTestDataset(test_claims, evidences)
    test_loader_clc = DataLoader(test_set_clc, batch_size=loader_batch_size, num_workers=loader_worker_num)

    net_clc = bclc.clc_pipeline(train_claims, dev_claims, evidences)

    claim_labels = bclc.decide_claim_labels(net_clc, test_loader_clc, gpu)
    bclc.extract_claim_evi_labels(test_claims, claim_labels, output_filename)  # This function saves the final predictions to the destination file


if __name__ == '__main__':
    print('Start running the pipeline...')