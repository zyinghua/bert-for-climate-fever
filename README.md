Original Paper can be found in:
<https://github.com/zyinghua/bert_for_climate_fever/blob/main/paper/A_BERT_based_model_for_Climate_Fact_Extraction_and_Claim_Verification.pdf>

# [Automated Fact Checking For Climate Science Claims] Implementation

This file aims to provide instructions and guides to run the implementation code provided that
as an entire system, produces the outcomes for Automated Fact Checking for Climate Science Claims.

### Table of contents

-   [1. Before you read the rest](#1-before-you-read-the-rest)
-   [2. Environment Initialisation](#2-environment-initialisation)
-   [3. Running Instruction](#3-running-instruction)
-   [4. Running the evidence retrieval training part](#4-evidence-retrival-training-part)
-   [5. Running the claim label prediction training part](#5-claim-label-prediction-training-part)
-   [6. Running the final prediction part](#6-prediction-part)
-   [7. Improvements](#7-improvements)

## 1. Before you read the rest

-   The code has been provided in two forms, firstly the ipynb file developed on Colab, and second is the python files as a locally-stored version containing the exact same content, with additionally also some code that has been used to experiment various aspects of the pipeline, but which did not turn to have a better outcome and therefore is deprecated after, although remained for archive, demonstration, reuse, proof and integrity purposes. I would suggest try on the Colab file rather than the python files, because some settings are considered in terms of the Colab environment, like the cuda GPU processing, that although the same code has been retained in the python files, but such as .cuda(0) commands may not be applicable locally. As well as the time to run locally is not experimented and is subjective to local computing resources. Also, the ipynb has a better partition declaration of the code, and the content below is mainly discussed based on the Colab environment, as assumed Colab is preferred based on the assignment information.

-   The code is expected to run with the GPU settings (GPU as the Hard accelerator in the colab runtime Notebook settings). Given GPU is enabled, it is expected to finish running within several hours on the basic GPU type (T4), but is never experimented and is solely based on the runtime difference observed between GPU type of T4 and GPU type of V100, that it generally takes within 3 hours with GPU type of V100, for the entire basic project pipeline (without Hard Negative Mining enabled, etc. Which there will be instructions to tell which to execute). Since all the experiments and work are done on Colab and based on the Colab Pro version, mainly on the V100 GPU type, so there might be discrepancy on the computational time cost when running the code if you are using the basic Colab version, which may depend on the Google server sometimes.

-   Since randomness is intergrated in some parts in the pipeline, on top of MLP layer parameters random initialisation, including the first training phrase of the evidence retrieval model which takes random negative evidences (in the range of the top TF-IDF cosine similar evidences given a claim) as training instances, and instances are randomly shuffled after pre-processing, therefore the results generated can be different (e.g., referring to the history test set performance on Codalab) while randomness is propagated through the entire pipeline, more or less, but shouldn't be significant.

-   The program generates some files along the way, in which you may only be interested in the final production file (i.e., test-claims-predictions.json).

-   **For the colab ipynb file, after setting up the environment discussed below, running the entire program is just as easy as clicking the running buttons for the indicated cells (mentioned below) from top to bottom sequentially. Apart from the instruction, the rest gives a brief intro of what each part does.**

## 2. Environment Initialisation

If run on Colab, please upload the relevant data files to a Google Drive folder under the account which will be used to run the Colab ipynb file, and make sure to specify the paths to the relevant files, as follows:

In the intialisation part of the ipynb file, at the second last cell of that part, where paths are specified according to my cases, please change the **`path_prefix`** variable (which is also indicated in the code comment) to where you can put and access the files. The pattern of the path looks like this: "/content/drive/MyDrive/`<repository path>`"

Meanwhile, please put all the data (train, dev, test) into a folder called 'project-data' (same as given), and put this folder in the path you specified as asked above.

## 3. Running Instruction

There are a few sections in the ipynb file, namely: 1. Initialisation, 2. Evidence Retrieval - Function Declarations, 3. Evidence Retrieval - Training 4. Evidence Retrieval Baseline Model (TFIDF) 5. Claim Label Classification - Function Declarations 6. Claim Label Classfication - Training 7. Claim Label Classfication - Zero R Baseline 8. Predict evidences and labels for test claims. 9. [Optional] Evidence Retrieval - Hard Negative Mining 10. [Optional] Claim Label Classification - Function Declarations | Alternative Version: Evidence Concatenation

Briefly speaking, You only need to run 1. [Initialisation], 2. [Evidence Retrieval - Function Declarations], 3. [Evidence Retrieval - Training], 5. [Claim Label Classification - Function Declarations], 6. [Claim Label Classification - Training] and 8. [Predict evidences and labels for test claims], in the exact same order.

Why not the others? :

Others are not used in the final pipeline due to either are considered as alternatives or relative less effective at this stage.

Others like baseline models have no relationship with the formal pipeline and are solely for testing and comparison purposes.

## 4. Evidence Retrieval Training Part

In the ipynb file (Once after the Initialisation part has been executed):

-   Evidence Retrieval Function Declarations: This part declares all the necessary functions for the evidence retrieval pipeline.

-   Evidence Retrieval - Training: This part initialises the datasets and dataloaders for train and dev (combined for the final version), which under the hood pre-filters the evidences based on the TF-IDF cosine similarity for each claim. And then train the BERT based model for 1 epoch, with all positive evidences and pos_num \* \<sample ratio\> number of negative evidences for each train claim.

-   [Optional, not in the final pipeline] Perform Hard Negative Mining: This part first loads the best performing model from the previous part, and then used to predict the train claims. The purpose of predicting train claims is to find those hard to distinguish negative evidences for each claim of the previous model, then will be used to help the model learn later. So for each claim, it will find the same number of negative evidences that the previous model had above the indicated confidence/probability of they are positives (considered as hard negatives), then record in a dictionary and break if enough hard negative evidences found, and proceed to next claim until done.

-   [Optional, Not in the final pipeline] Evidence Retrieval - Train on Hard Negative Evidences from HNM: So once hard negative evidences of all claims obtained for the previous model, we then train the model by feeding in the all the positive evidences and all hard negative evidences with some random negative evidences (num = half of num of positive evidences for a given claim), for 13 epochs, with smaller learning rate.

-   Evidence Retrieval Baseline Model: This is just a TF-IDF with cosine similarity based model, that selects the top 5 evidences that are most cosine similar of the TF-IDF representation of a given claim.

In the python file:

-   The bert_er.py corresponds to what are discussed above, content is pretty much the same.

## 5. Claim label Prediction Training Part

In the ipynb file:

-   Claim Label Classification - Function Declarations: This part declares all the necessary functions for the claim label prediction pipeline.

-   Claim Label Classification Training: This part defines the datasets and dataloaders for train and dev, which unrolls claim evidences into pairs with the corresponding labels for all the associative pairs as the label of their claim. The label is either SUPPORTS, REFUTES or NOT_ENOUGH_INFO, as DISPUTED is not pure (can have different labels for each individual pair). Then it trains the BERT based model for 9 epochs. Final label for each claim is selected based on majority voting for the final pipeline submitted.

In the python files:
Worthing mentioning that bert_clc.py, bert_clcconcatevi.py and bert_clcaggrlayer.py are all for claim label classification but using different approaches. bert_clcconcatevi.py concatnates all evidences of a claim into one phrase and direct forward into the BERT model with the associated claim for final label prediction. bert_clcaggrlayer.py inserts another aggregation layer after each claim evidence pair prediction, then takes the probabilities of each class, concatenate, and aggregate for claim final label. bert_clc.py on top of predicting each claim evidence pair, aggreagte the results with 3 different methods: weighted voting, majority voting and rule aggregation. bert_clc.py with majority voting is used for the final version based on the best performance observed during experiment, based on the previous pipeline details.

## 6. Prediction Part

This part loads the respective best performing models for evidence retrieval and claim label prediction trained in the previous steps, with initialisation of test set and test loader, make predictions then saves the results in the file named "test-claims-predictions.json". The test set is preprocessed to select the top 1000 TF-IDF cosine similar evidences regarding a claim that will then be forwarded for the BERT model to further process.

## 7. Improvements
Have to say this implementation was mainly for learning and testing purposes, there can be quite a lot of improvements, or better alternative strategies for this topic, for example examination of BM25, reranking systems, even using community models, etc. any proper decisions at different levels could possibly significantly improve the performance of the system. This project was within a university setting, therefore further improvements have not yet been implemented although considered, may happen in the future.
