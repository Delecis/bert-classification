# -*- coding: utf-8 -*-

import os
import argparse
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import (RobertaTokenizer, RobertaConfig,
                          RobertaForSequenceClassification,
                          get_linear_schedule_with_warmup,
                          AdamW)
from transformers import XLNetTokenizer, XLNetForSequenceClassification
from MYBERT.model import *
from transformers import AutoModel, AutoTokenizer
import numpy as np
from utils.predict_utils import create_dataset, vote, write_result, mean
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def split_(str_):
    str_list = str_.split(',')
    str_list = [element for element in str_list if len(element) > 0]
    return str_list


def predict(args, model, dataset):
    samper = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=samper, batch_size=args.batch_size)
    preds_logits = None
    model.eval()
    for batch in tqdm(dataloader, desc="Predicting"):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1]
                      # 'token_type_ids': batch[2]
                      }

            outputs = model(**inputs)
            # logits = outputs[0]
            logits = outputs[1]
        if preds_logits is None:
            preds_logits = logits.detach().cpu().numpy()
        else:
            preds_logits = np.append(preds_logits, logits.detach().cpu().numpy(), axis=0)

    predictions = np.argmax(preds_logits, axis=1)
    return predictions, preds_logits


def all_predict(args, model_paths):
    predictions = []
    preds_logits = []
    ids = None
    for model_path in model_paths:
        model = RobertaForClass_R.from_pretrained(model_path)
        # model = BertForSequenceClassification.from_pretrained(model_path)

        model.to(args.device)
        tokenizer = RobertaTokenizer.from_pretrained(model_path,
                                                  do_lower_case=args.do_lower_case)
        # tokenizer = AutoModel.from_pretrained("vinai/bertweet-large",do_lower_case=args.do_lower_case)
        dataset, ids = create_dataset(args.predict_file, tokenizer, args.max_seq_length)
        prediction, preds_logit = predict(args, model, dataset)
        predictions.append(prediction)
        preds_logits.append(preds_logit)
    return predictions, preds_logits, ids


def main():
    parser = argparse.ArgumentParser()

    # ???????????????????????????????????????????????????????????????k???fold???????????????
    parser.add_argument(
        "--fold_model_paths",
        default="",
        type=str,
        help="Path to pre-trained models",
    )

    parser.add_argument(
        "--vote_model_paths",
        default='../user_data/tmp_data/checkpoints/roberta_wwm_large,../user_data/tmp_data/checkpoints/ernie,../user_data/tmp_data/checkpoints/nezha-cn-base,../user_data/tmp_data/checkpoints/roberta_large_pair,../user_data/tmp_data/checkpoints/roberta_tiny_pair',
        type=str,
        help="Path to pre-trained models",
    )
    ###
    # ????????????
    parser.add_argument(
        "--predict_file",
        ###default='/tcdata/test.csv',
        ### ?????????????????????
        default='../data/Dataset/test.csv',
        type=str,
        help=".",
    )
    # ??????????????????
    parser.add_argument(
        "--predict_result_file",
        default='result.csv',
        type=str,
        help=".",
    )

    parser.add_argument(
        "--max_seq_length",
        default=256,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--do_lower_case", default=True, type=bool, help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--batch_size", default=32, type=int, help="Batch size for prediction.",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    args = parser.parse_args()
    # Setup CUDA, GPU & distributed training

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device

    fold_model_paths = split_(args.fold_model_paths)
    ids = None
    fold_predictions = []

    # ?????????????????????logits?????????
    if len(fold_model_paths) > 0:
        for fold_model_path in fold_model_paths:
            list_dir = os.listdir(fold_model_path)
            model_dirs = [os.path.join(fold_model_path, dir_) for dir_ in list_dir]
            _, fold_logit, ids = all_predict(args, model_dirs)
            fold_logit = mean(fold_logit)
            fold_prediction = np.argmax(fold_logit, axis=1)
            fold_predictions.append(fold_prediction)

    # ?????????????????????
    vote_model_paths = split_(args.vote_model_paths)
    vote_predictions = []
    if len(vote_model_paths) > 0:
        for vote_model_path in vote_model_paths:
            list_dir = os.listdir(vote_model_path)
            model_dirs = [os.path.join(vote_model_path, dir_) for dir_ in list_dir]
            vote_prediction, _, ids = all_predict(args, model_dirs)
            vote_predictions.append(vote(vote_prediction))

    # ??????????????????????????????????????????????????????
    predictions = vote(fold_predictions + vote_predictions)

    write_result(args.predict_result_file, ids, predictions)


if __name__ == '__main__':
    main()
