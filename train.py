# -*- coding: utf-8 -*-
#；lklll;ll
import argparse
# import logging
# import os
import random
import numpy as np
import csv
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import logging
from MYBERT.model import *
# from transformers import AutoModel, AutoTokenizer
# AutoModel.from_pretrained("vinai/bertweet-base")
# AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)


# logging.basicConfig(level=logging.ERROR)
from transformers import (BertTokenizer, BertConfig,
                          BertForSequenceClassification,
                          get_linear_schedule_with_warmup,
                          AdamW)
from transformers import (RobertaTokenizer, RobertaConfig,
                          RobertaForSequenceClassification,
                          get_linear_schedule_with_warmup,
                          AdamW)

from transformers import (XLNetTokenizer, XLNetConfig,XLNetForSequenceClassification,
                          get_linear_schedule_with_warmup,
                          AdamW)


from utils.data_utils import (PROCESSORS, multi_classification_convert_examples_to_dataset,
                               compute_metrics)
from utils.adv_utils import *

logger = logging.getLogger(__name__)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def load_and_cache_examples(args, tokenizer, set_type='train'):
    if args.local_rank not in [-1, 0] and set_type == 'train':
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = PROCESSORS[args.task_name]()
    logger.info("Creating {} dataset of {} task".format(set_type, args.task_name))
    examples = processor.get_examples(args.data_dir, set_type)
    label_list = processor.get_labels()
    if isinstance(label_list, dict):
        label_list = list(label_list.keys())
    dataset = multi_classification_convert_examples_to_dataset(
        examples,
        tokenizer,
        max_length=args.max_seq_length,
        label_list=label_list,
        pad_on_left=False,
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
        threads=args.threads,
        set_type=set_type
    )
    return dataset


def train(args, tokenizer, model):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_dataset = load_and_cache_examples(args, tokenizer, set_type='train')
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    eval_dataset = None
    if args.do_eval_during_train:
        eval_dataset = load_and_cache_examples(args, tokenizer, set_type='dev')

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    args.warmup_steps = int(t_total * args.warmup_rate)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    fgm = None
    if args.adv_type == 'fgm':
        fgm = FGM(model)

    pgd = None
    if args.adv_type == 'pgd':
        pgd = PGD(model)
        pgd_k = 3

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_acc = 0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      # 'token_type_ids': batch[2],
                      # 'labels': batch[3]}
                      'labels': batch[2]}

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                # 下行代码用来防止梯度消失，直接设定了梯度截断的阈值：https://blog.csdn.net/Mikeyboi/article/details/119522689
                # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)

            if args.adv_type == 'fgm':
                fgm.attack()  ##对抗训练
                loss_adv = model(**inputs)[0]
                loss_adv.backward()
                fgm.restore()


            if args.adv_type == 'pgd':
                pgd.backup_grad()
                # 对抗训练
                for t in range(pgd_k):
                    pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
                    # 原代码，我觉得这里写的不对, 不需要清零梯度和恢复梯度，只需逐渐累加梯度就行了，在超球形内缓步走
                    # 再次review发现是对的，因为前面几步的梯度不要累计进去，所有步数走完的梯度和原梯度累计即可
                    if t != pgd_k-1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                    #loss_adv = model(**inputs)[0]
                    loss_adv = model(**inputs)[0].mean()
                    #loss_adv = loss_fn(logits,label_t)
                    loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                pgd.restore() # 恢复embedding参数

            tr_loss += loss.item()
            epoch_iterator.set_description("loss {}".format(round(loss.item(), 4)))
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and args.do_eval_during_train and (
                        global_step % args.logging_steps == 0 or (global_step + 1) == t_total) and eval_dataset:
                    eval_result, _ = evaluate(args, model, eval_dataset, tokenizer)
                    output_dir = args.output_dir
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    current_acc = eval_result['accuracy']

                    logger.info("  best accuracy : {}".format(best_acc))
                    logger.info("  current accuracy : {}".format(current_acc))
                    logger.info("  current step : {}".format(global_step))
                    logger.info("  ")
                    for k in eval_result.keys():
                        logger.info("  eval {} : {}".format(k, eval_result[k]))
                    if current_acc > best_acc:
                        best_acc = current_acc
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

                        logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    return global_step


def evaluate(args, model, eval_dataset=None, tokenizer=None):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    if not eval_dataset and tokenizer:
        eval_dataset = load_and_cache_examples(args, tokenizer, set_type='dev')
    if not eval_dataset:
        raise ValueError('The eval or test dataset can not be None')

    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      # 'token_type_ids': batch[2],
                      # 'labels': batch[3]}
                      'labels': batch[2]}
            # 把labels:batch[3]试试，不过前面的都得重新修改就是了，效果应该没什么区别
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    result = compute_metrics(out_label_ids, preds, average='micro')
    result['eval_loss'] = round(eval_loss, 4)
    return result, preds


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--adv_type', default='pgd', type=str, choices=['fgm','pgd', None])


    parser.add_argument(
        "--model_name_or_path",
        default='../model/roberta_tiny_pair',
        type=str,
        help="Path to pre-trained model ",
    )

    parser.add_argument(
        "--data_dir",
        default='../user_data/tmp_data/Kfold/4',
        type=str,
        help="Path to data ",
    )
    parser.add_argument(
        "--task_name",
        default='pair',
        type=str,
        help="The name of the task to train selected in the list: " + ", ".join(PROCESSORS.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default='../user_data/tmp_data/checkpoints/roberta_tiny_pair/4',
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type  =str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--max_seq_length",
        default=256,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true",default=True, help="Whether to run training.")
    parser.add_argument("--do_eval", default=True, help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--do_eval_during_train", action="store_true", default=True, help="Run evaluation during training at each logging step.",
    )
    ###

    parser.add_argument(
        "--do_lower_case", default=False, type=bool, help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=32, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    # 模型：lr; roberta_wwm_large:2e-5; roberta_pair:5e-6; ernie:4e-5;
    #parser.add_argument("--learning_rate", default=4e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_rate", default=0.1, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")

    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", type=bool, default=True, help="Overwrite the content of the output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--threads", type=int, default=10, help="multiple threads for converting example to features")

    args = parser.parse_args()

    # args.output_dir = os.path.join(args.output_dir, args.task_name)
    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )
    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in PROCESSORS:
        raise ValueError("Task not found: %s" % (args.task_name))

    processor = PROCESSORS[args.task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        config = RobertaConfig.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            num_labels=num_labels
        )

        tokenizer = RobertaTokenizer.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            do_lower_case=args.do_lower_case,
        )
        # tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-large",do_lower_case=args.do_lower_case)

        model = RobertaForClass_R.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config
        )

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        model.to(args.device)
        global_step = train(args, tokenizer, model)
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        # Save the trained model and the tokenizer
        if (args.local_rank == -1 or torch.distributed.get_rank() == 0) and (
                not args.do_eval_during_train):
            output_dir = args.output_dir
            if not os.path.exists(output_dir) and args.local_rank in [-1, 0]:
                os.makedirs(output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    if args.do_eval and args.local_rank in [-1, 0]:
        output_dir = args.output_dir
        model = RobertaForClass_R.from_pretrained(output_dir)
        model.to(args.device)
        # tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-large",do_lower_case=args.do_lower_case)
        tokenizer = RobertaTokenizer.from_pretrained(output_dir,
                                                  do_lower_case=args.do_lower_case)
        result, _ = evaluate(args, model, tokenizer=tokenizer)
        output_eval_file = os.path.join(output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for k, v in result.items():
                logger.info("  {} : {}".format(k, v))
                writer.write("{} : {}\n".format(k, v))

if __name__ == '__main__':
   # --do_train,

    main()
