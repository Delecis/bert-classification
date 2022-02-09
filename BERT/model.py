
from transformers.activations import get_activation

from .module import FocalLoss

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch
import random
import os
from torch import nn, optim
import torch.nn.functional as F
from transformers.activations import get_activation
from transformers import BertTokenizer, AdamW, BertModel, BertPreTrainedModel, BertConfig, \
    get_linear_schedule_with_warmup, XLNetModel, XLNetTokenizer, XLNetConfig, ElectraModel, ElectraConfig, ElectraTokenizer, \
    RobertaTokenizer, RobertaModel, RobertaConfig,RobertaPreTrainedModel


# class BertForClass(nn.Module): # 即最后一层取平均并与cls拼接
#     def __init__(self, config, args):
#         super(BertForClass, self).__init__()
#         self.args = args
#         self.n_classes = args.num_class
#
#         config_json = 'config.json' if os.path.exists(args.model_path + 'config.json') else 'config.json'
#         self.bert_config = BertConfig.from_pretrained(args.model_path + config_json,
#                                                                  output_hidden_states=True)
#         self.bert_model = BertModel.from_pretrained(args.model_path, config=self.bert_config)
#         self.isDropout = True if 0 < args.dropout < 1 else False
#         self.dropout = nn.Dropout(p=args.dropout)
#         self.classifier = nn.Linear(self.bert_config.hidden_size * 2, self.n_classes)
#
#     def forward(self, input_ids, attention_mask, token_type_ids,labels):
#         outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids,attention_mask=attention_mask)
#
#         sequence_output = outputs[0]
#         pooler_output = outputs[1]
#         hidden_states = outputs[2]
#         seq_avg = torch.mean(sequence_output, dim=1)
#         # 取sequence_output均值和pooler_output拼接
#         concat_out = torch.cat((seq_avg, pooler_output), dim=1)
#         if self.isDropout:
#             concat_out = self.dropout(concat_out)
#         logits = self.classifier(concat_out)
#
#         loss = None
#         if labels is not None:
#             if self.config.problem_type is None:
#                 if self.num_labels == 1:
#                     self.config.problem_type = "regression"
#                 elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
#                     self.config.problem_type = "single_label_classification"
#                 else:
#                     self.config.problem_type = "multi_label_classification"
#
#             if self.config.problem_type == "regression":
#                 loss_fct = MSELoss()
#                 if self.num_labels == 1:
#                     loss = loss_fct(logits.squeeze(), labels.squeeze())
#                 else:
#                     loss = loss_fct(logits, labels)
#             elif self.config.problem_type == "single_label_classification":
#             #loss_fct = CrossEntropyLoss()
#                 loss_fct = FocalLoss(self.num_labels,alpha=0.25,gamma=2.0,size_average=True)
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             elif self.config.problem_type == "multi_label_classification":
#                 #loss_fct = BCEWithLogitsLoss()
#                 loss_fct = FocalLoss(self.num_labels,alpha=0.25,gamma=2.0,size_average=True)
#                 loss = loss_fct(logits, labels)
#
#             output = (logits,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output


# 最后返回的是经过线性层的Logit（之后进行计算loss和softmax）




# 改版
class BertForClass(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size * 2, config.num_labels)
        #self.classifier = nn.Linear(config.hidden_size * 4, config.num_labels)
        self.init_weights()

        # config_json = 'bert_config.json' if os.path.exists(config.model_path + 'config.json') else 'config.json'
        # self.bert_config = CONFIGS[config.model].from_pretrained(config.model_path + config_json,
        #                                                          output_hidden_states=True)
        # self.bert_model = MODELS[config.model].from_pretrained(config.model_path, config=self.bert_config)
        # self.isDropout = True if 0 < config.dropout < 1 else False
        # self.dropout = nn.Dropout(p=config.dropout)
        # self.classifier = nn.Linear(self.bert_config.hidden_size * 2, self.n_classes)
    # def forward(
    #         self,
    #         input_ids=None,
    #         attention_mask=None,
    #         token_type_ids=None,
    #         position_ids=None,
    #         head_mask=None,
    #         inputs_embeds=None,
    #         labels=None,
    #         output_attentions=None,
    #         output_hidden_states=None,
    #         return_dict=None,
    # ):
    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids,
                                                                        attention_mask=attention_mask)
        #return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids,
        #     position_ids=position_ids,
        #     head_mask=head_mask,
        #     inputs_embeds=inputs_embeds,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )
        sequence_output = outputs[0]
        pooled_output = outputs[1]
        #hidden_states = outputs[2]
#######################
        # output = torch.cat(
        #     (hidden_states[-1][:, 0], hidden_states[-2][:, 0], hidden_states[-3][:, 0], hidden_states[-4][:, 0]), dim=1)
        #
        # output = self.dropout(output)
        # logits = self.classifier(output)
############################
        # outputs = (logits,) + outputs[2: ]
        seq_avg = torch.mean(sequence_output, dim=1)
        concat_out = torch.cat((seq_avg, pooled_output), dim=1)

        concat_out = self.dropout(concat_out)
        logits = self.classifier(concat_out)
        outputs = (logits,) + outputs[2: ]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                #loss_fct = CrossEntropyLoss()
                loss_fct = FocalLoss(self.num_labels)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        outputs = (loss,) + outputs
        return outputs


def compute_kl_loss(p, q):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2

    return loss



class RobertaForClass_RMM(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config,add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        #self.rdrop_coef = 0.0
        self.rdrop_coef = 4.0
        # self.rdropout = nn.Sequential(nn.Dropout(0.3),nn.Dropout(0.3))

        # self.dropout = nn.Dropout(classifier_dropout)
        # self.classifier = nn.Linear(config.hidden_size * 2, config.num_labels)
        self.classifier = RobertaClassificationHead_R(config)
        self.init_weights()


    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids=input_ids,
                               attention_mask=attention_mask)#,output_hidden_states=True)


        ########################原模型#####################
        # pooled_output = outputs[1]
        # pooled_output = self.rdropout(pooled_output)
        # logits = self.classifier(pooled_output)
        # outputs = (logits,) + (pooled_output,)
        ######################改了CLS的模型##################
        sequence_output = outputs[0]
        # pooled_output = outputs[1].mean()
        seq_avg = torch.mean(sequence_output, dim=1)
        # concat_out = torch.cat((seq_avg, pooled_output), dim=1)
        #########这里换为rdropout
        # concat_out = self.rdropout(concat_out)
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2: ]


        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                #loss_fct = CrossEntropyLoss()
                loss_fct = FocalLoss(self.num_labels)
                # r-dropout
                if self.rdrop_coef > 0:
                    outputs1 = self.roberta(input_ids=input_ids,
                                            attention_mask=attention_mask)
                    ##################原版##########################
                    #sequence_output1 = outputs[0]
                    # pooled_output1 = outputs1[1]
                    # pooled_output1 = self.rdropout(pooled_output1)
                    # logits1 = self.classifier(pooled_output1)
                    ##################替换为cls版####################
                    sequence_output1 = outputs1[0]
                    # pooled_output1 = outputs1[1]
                    # seq_avg = torch.mean(sequence_output1, dim=1)
                    # concat_out1 = torch.cat((seq_avg, pooled_output1), dim=1)
                    # concat_out1 = self.rdropout(concat_out1)
                    logits1 = self.classifier(sequence_output1)

                    ce_loss = 0.5 * (loss_fct(logits.view(-1, self.num_labels), labels.view(-1)) +
                                     loss_fct(logits1.view(-1, self.num_labels), labels.view(-1)))
                    kl_loss = compute_kl_loss(logits, logits1)
                    loss = ce_loss + self.rdrop_coef * kl_loss
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                #loss_fct = FocalLoss(self.num_labels)
                #loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        outputs = (loss,) + outputs
        return outputs



class BertForClass_R(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        # classifier_dropout = (
        #     config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        # )
        classifier_dropout = config.hidden_dropout_prob

        #self.rdrop_coef = 0.0
        self.rdrop_coef = 4.0
        self.rdropout = nn.Sequential(nn.Dropout(0.3),nn.Dropout(0.3))

        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size * 2, config.num_labels)
        self.init_weights()


    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask)#,output_hidden_states=True)


        ########################原模型#####################
        # pooled_output = outputs[1]
        # pooled_output = self.rdropout(pooled_output)
        # logits = self.classifier(pooled_output)
        # outputs = (logits,) + (pooled_output,)
        ######################改了CLS的模型##################
        sequence_output = outputs[0]
        pooled_output = outputs[1]
        seq_avg = torch.mean(sequence_output, dim=1)
        concat_out = torch.cat((seq_avg, pooled_output), dim=1)
        #########这里换为rdropout
        concat_out = self.rdropout(concat_out)
        logits = self.classifier(concat_out)
        outputs = (logits,) + outputs[2: ]


        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            # elif self.config.problem_type == "single_label_classification":
            else:
                #loss_fct = CrossEntropyLoss()
                loss_fct = FocalLoss(self.num_labels)
                # r-dropout
                if self.rdrop_coef > 0:
                    outputs1 = self.bert(input_ids=input_ids, token_type_ids=token_type_ids,
                                        attention_mask=attention_mask)
                    ##################原版##########################
                    #sequence_output1 = outputs[0]
                    # pooled_output1 = outputs1[1]
                    # pooled_output1 = self.rdropout(pooled_output1)
                    # logits1 = self.classifier(pooled_output1)
                    ##################替换为cls版####################
                    sequence_output1 = outputs1[0]
                    pooled_output1 = outputs1[1]
                    seq_avg = torch.mean(sequence_output1, dim=1)
                    concat_out1 = torch.cat((seq_avg, pooled_output1), dim=1)
                    concat_out1 = self.rdropout(concat_out1)
                    logits1 = self.classifier(concat_out1)

                    ce_loss = 0.5 * (loss_fct(logits.view(-1, self.num_labels), labels.view(-1)) +
                                     loss_fct(logits1.view(-1, self.num_labels), labels.view(-1)))
                    kl_loss = compute_kl_loss(logits, logits1)
                    loss = ce_loss + self.rdrop_coef * kl_loss
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                #loss_fct = FocalLoss(self.num_labels)
                #loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            # elif self.config.problem_type == "multi_label_classification":
            #
            #     loss_fct = BCEWithLogitsLoss()
            #     loss = loss_fct(logits, labels)

        outputs = (loss,) + outputs
        return outputs


class RobertaClassificationHead_R(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # classifier_dropout = (
        #     config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        # )
        # classifier_dropout = 0.1
        # self.dropout = nn.Dropout(classifier_dropout)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x



class RobertaForClass_R(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config,add_pooling_layer=False)
        # classifier_dropout = (
        #     config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        # )
        #self.rdrop_coef = 0.0
        self.rdrop_coef = 4.0
        # self.rdropout = nn.Sequential(nn.Dropout(0.3),nn.Dropout(0.3))

        # self.dropout = nn.Dropout(classifier_dropout)
        # self.classifier = nn.Linear(config.hidden_size * 2, config.num_labels)
        self.classifier = RobertaClassificationHead_R(config)
        self.init_weights()


    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids=input_ids,
                            attention_mask=attention_mask)#,output_hidden_states=True)


        ########################原模型#####################
        # pooled_output = outputs[1]
        # pooled_output = self.rdropout(pooled_output)
        # logits = self.classifier(pooled_output)
        # outputs = (logits,) + (pooled_output,)
        ######################改了CLS的模型##################
        sequence_output = outputs[0]
        # pooled_output = outputs[1].mean()
        seq_avg = torch.mean(sequence_output, dim=1)
        # concat_out = torch.cat((seq_avg, pooled_output), dim=1)
        #########这里换为rdropout
        # concat_out = self.rdropout(concat_out)
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2: ]


        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                #loss_fct = CrossEntropyLoss()
                loss_fct = FocalLoss(self.num_labels)
                # r-dropout
                if self.rdrop_coef > 0:
                    outputs1 = self.roberta(input_ids=input_ids,
                                         attention_mask=attention_mask)
                    ##################原版##########################
                    #sequence_output1 = outputs[0]
                    # pooled_output1 = outputs1[1]
                    # pooled_output1 = self.rdropout(pooled_output1)
                    # logits1 = self.classifier(pooled_output1)
                    ##################替换为cls版####################
                    sequence_output1 = outputs1[0]
                    # pooled_output1 = outputs1[1]
                    # seq_avg = torch.mean(sequence_output1, dim=1)
                    # concat_out1 = torch.cat((seq_avg, pooled_output1), dim=1)
                    # concat_out1 = self.rdropout(concat_out1)
                    logits1 = self.classifier(sequence_output1)

                    ce_loss = 0.5 * (loss_fct(logits.view(-1, self.num_labels), labels.view(-1)) +
                                     loss_fct(logits1.view(-1, self.num_labels), labels.view(-1)))
                    kl_loss = compute_kl_loss(logits, logits1)
                    loss = ce_loss + self.rdrop_coef * kl_loss
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                #loss_fct = FocalLoss(self.num_labels)
                #loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        outputs = (loss,) + outputs
        return outputs
# 原版
# class BertForClass(nn.Module):
#     def __init__(self, config):
#         super(BertForClass, self).__init__()
#         self.n_classes = config.num_class
#
#         config_json = 'bert_config.json' if os.path.exists(config.model_path + 'bert_config.json') else 'config.json'
#         self.bert_config = CONFIGS[config.model].from_pretrained(config.model_path + config_json,
#                                                                  output_hidden_states=True)
#         self.bert_model = MODELS[config.model].from_pretrained(config.model_path, config=self.bert_config)
#         self.isDropout = True if 0 < config.dropout < 1 else False
#         self.dropout = nn.Dropout(p=config.dropout)
#         self.classifier = nn.Linear(self.bert_config.hidden_size * 2, self.n_classes)
#
#     def forward(self, input_ids, input_masks, segment_ids):
#         output = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids,
#                                  attention_mask=input_masks)
#         sequence_output = output[0]
#         pooler_output = output[1]
#         hidden_states = output[2]
#         seq_avg = torch.mean(sequence_output, dim=1)
#         concat_out = torch.cat((seq_avg, pooler_output), dim=1)
#         if self.isDropout:
#             concat_out = self.dropout(concat_out)
#         logit = self.classifier(concat_out)
#         return logit



# class BertForClass_MultiDropout(nn.Module):
#     def __init__(self, config):
#         super(BertForClass_MultiDropout, self).__init__()
#         self.n_classes = config.num_class
#         config_json = 'bert_config.json' if os.path.exists(config.model_path + 'bert_config.json') else 'config.json'
#         self.bert_config = CONFIGS[config.model].from_pretrained(config.model_path + config_json,
#                                                                  output_hidden_states=True)
#
#         self.bert_model = MODELS[config.model].from_pretrained(config.model_path, config=self.bert_config)
#         self.multi_drop = 5
#         self.multi_dropouts = nn.ModuleList([nn.Dropout(config.dropout) for _ in range(self.multi_drop)])
#         self.classifier = nn.Linear(self.bert_config.hidden_size * 2, self.n_classes)
#
#     def forward(self, input_ids, input_masks, segment_ids):
#         sequence_output, pooler_output, hidden_states = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids,
#                                                                         attention_mask=input_masks)
#         seq_avg = torch.mean(sequence_output, dim=1)
#         concat_out = torch.cat((seq_avg, pooler_output), dim=1)
#         for j, dropout in enumerate(self.multi_dropouts):
#             if j == 0:
#                 logit = self.classifier(dropout(concat_out)) / self.multi_drop
#             else:
#                 logit += self.classifier(dropout(concat_out)) / self.multi_drop
#
#         return logit
#
# class BertLastTwoCls(nn.Module):
#     def __init__(self, config):
#         super(BertLastTwoCls, self).__init__()
#         self.n_classes = config.num_class
#         config_json = 'bert_config.json' if os.path.exists(config.model_path + 'bert_config.json') else 'config.json'
#         self.bert_config = CONFIGS[config.model].from_pretrained(config.model_path + config_json,
#                                                                  output_hidden_states=True)
#         self.bert_model = MODELS[config.model].from_pretrained(config.model_path, config=self.bert_config)
#         self.isDropout = True if 0 < config.dropout < 1 else False
#         self.dropout = nn.Dropout(p=config.dropout)
#         self.classifier = nn.Linear(self.bert_config.hidden_size, config.num_class)
#
#     def forward(self, input_ids, input_masks, segment_ids):
#         sequence_output, pooler_output, hidden_states = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids,
#                                                                         attention_mask=input_masks)
#         logit = self.classifier(pooler_output)
#
#         return logit
#
#
# class BertLastCls(nn.Module):
#     def __init__(self, config):
#         super(BertLastCls, self).__init__()
#         self.n_classes = config.num_class
#         config_json = 'bert_config.json' if os.path.exists(config.model_path + 'bert_config.json') else 'config.json'
#         self.bert_config = CONFIGS[config.model].from_pretrained(config.model_path + config_json,
#                                                                  output_hidden_states=True)
#         self.bert_model = MODELS[config.model].from_pretrained(config.model_path, config=self.bert_config)
#         self.isDropout = True if 0 < config.dropout < 1 else False
#         self.dropout = nn.Dropout(p=config.dropout)
#         self.classifier = nn.Linear(self.bert_config.hidden_size, config.num_class)
#
#     def forward(self, input_ids, input_masks, segment_ids):
#         output = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids,
#                                  attention_mask=input_masks)
#         sequence_output = output[0]
#         pooler_output = output[1]
#         hidden_states = output[2]
#
#         if self.isDropout:
#             output = self.dropout(pooler_output)
#         logit = self.classifier(output)
#
#         return logit
#
# class BertLastTwoClsPooler(nn.Module):
#     def __init__(self, config):
#         super(BertLastTwoClsPooler, self).__init__()
#         self.n_classes = config.num_class
#         config_json = 'bert_config.json' if os.path.exists(config.model_path + 'bert_config.json') else 'config.json'
#         self.bert_config = CONFIGS[config.model].from_pretrained(config.model_path + config_json,
#                                                                  output_hidden_states=True)
#         self.bert_model = MODELS[config.model].from_pretrained(config.model_path, config=self.bert_config)
#         self.isDropout = True if 0 < config.dropout < 1 else False
#         self.dropout = nn.Dropout(p=config.dropout)
#         self.classifier = nn.Linear(self.bert_config.hidden_size * 3, config.num_class)
#
#     def forward(self, input_ids, input_masks, segment_ids):
#         sequence_output, pooler_output, hidden_states = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids,
#                                                                         attention_mask=input_masks)
#
#         output = torch.cat(
#             (pooler_output, hidden_states[-1][:, 0], hidden_states[-2][:, 0]), dim=1)
#         if self.isDropout:
#             output = self.dropout(output)
#         logit = self.classifier(output)
#
#         return logit
#
# class BertLastTwoEmbeddings(nn.Module):
#     def __init__(self, config):
#         super(BertLastTwoEmbeddings, self).__init__()
#         self.n_classes = config.num_class
#         config_json = 'bert_config.json' if os.path.exists(config.model_path + 'bert_config.json') else 'config.json'
#         self.bert_config = CONFIGS[config.model].from_pretrained(config.model_path + config_json,
#                                                                  output_hidden_states=True)
#         self.bert_model = MODELS[config.model].from_pretrained(config.model_path, config=self.bert_config)
#         self.isDropout = True if 0 < config.dropout < 1 else False
#         self.dropout = nn.Dropout(p=config.dropout)
#         self.classifier = nn.Linear(self.bert_config.hidden_size * 2, config.num_class)
#
#     def forward(self, input_ids, input_masks, segment_ids):
#         sequence_output, pooler_output, hidden_states = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids,
#                                                                         attention_mask=input_masks)
#
#         hidden_states1 = torch.mean(hidden_states[-1], dim=1)
#         hidden_states2 = torch.mean(hidden_states[-2], dim=1)
#
#         output = torch.cat(
#             (hidden_states1, hidden_states2), dim=1)
#         if self.isDropout:
#             output = self.dropout(output)
#         logit = self.classifier(output)
#
#         return logit
#
#
# class BertLastTwoEmbeddingsPooler(nn.Module):
#     def __init__(self, config):
#         super(BertLastTwoEmbeddingsPooler, self).__init__()
#         self.n_classes = config.num_class
#         config_json = 'bert_config.json' if os.path.exists(config.model_path + 'bert_config.json') else 'config.json'
#         self.bert_config = CONFIGS[config.model].from_pretrained(config.model_path + config_json,
#                                                                  output_hidden_states=True)
#         self.bert_model = MODELS[config.model].from_pretrained(config.model_path, config=self.bert_config)
#         self.isDropout = True if 0 < config.dropout < 1 else False
#         self.dropout = nn.Dropout(p=config.dropout)
#         self.classifier = nn.Linear(self.bert_config.hidden_size * 3, config.num_class)
#
#     def forward(self, input_ids, input_masks, segment_ids):
#         sequence_output, pooler_output, hidden_states = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids,
#                                                                         attention_mask=input_masks)
#
#         hidden_states1 = torch.mean(hidden_states[-1], dim=1)
#         hidden_states2 = torch.mean(hidden_states[-2], dim=1)
#
#         output = torch.cat(
#             (pooler_output, hidden_states1, hidden_states2), dim=1)
#         if self.isDropout:
#             output = self.dropout(output)
#         logit = self.classifier(output)
#
#         return logit
#
# class BertLastFourCls(nn.Module):
#     def __init__(self, config):
#         super(BertLastFourCls, self).__init__()
#         self.n_classes = config.num_class
#         config_json = 'bert_config.json' if os.path.exists(config.model_path + 'bert_config.json') else 'config.json'
#         self.bert_config = CONFIGS[config.model].from_pretrained(config.model_path + config_json,
#                                                                  output_hidden_states=True)
#         self.bert_model = MODELS[config.model].from_pretrained(config.model_path, config=self.bert_config)
#         self.isDropout = True if 0 < config.dropout < 1 else False
#         self.dropout = nn.Dropout(p=config.dropout)
#         self.classifier = nn.Linear(self.bert_config.hidden_size * 4, config.num_class)
#
#     def forward(self, input_ids, input_masks, segment_ids):
#         output = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids,
#                                  attention_mask=input_masks)
#         sequence_output = output[0]
#         pooler_output = output[1]
#         hidden_states = output[2]
#
#         output = torch.cat(
#             (hidden_states[-1][:, 0], hidden_states[-2][:, 0], hidden_states[-3][:, 0], hidden_states[-4][:, 0]), dim=1)
#         if self.isDropout:
#             output = self.dropout(output)
#         logit = self.classifier(output)
#
#         return logit
#
#
# class BertLastFourClsPooler(nn.Module):
#     def __init__(self, config):
#         super(BertLastFourClsPooler, self).__init__()
#         self.n_classes = config.num_class
#         config_json = 'bert_config.json' if os.path.exists(config.model_path + 'bert_config.json') else 'config.json'
#         self.bert_config = CONFIGS[config.model].from_pretrained(config.model_path + config_json,
#                                                                  output_hidden_states=True)
#         self.bert_model = MODELS[config.model].from_pretrained(config.model_path, config=self.bert_config)
#         self.isDropout = True if 0 < config.dropout < 1 else False
#         self.dropout = nn.Dropout(p=config.dropout)
#         self.classifier = nn.Linear(self.bert_config.hidden_size * 5, config.num_class)
#
#     def forward(self, input_ids, input_masks, segment_ids):
#         sequence_output, pooler_output, hidden_states = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids,
#                                                                         attention_mask=input_masks)
#
#         output = torch.cat(
#             (pooler_output, hidden_states[-1][:, 0], hidden_states[-2][:, 0], hidden_states[-3][:, 0], hidden_states[-4][:, 0]), dim=1)
#         if self.isDropout:
#             output = self.dropout(output)
#         logit = self.classifier(output)
#
#         return logit
#
# class BertLastFourEmbeddings(nn.Module):
#     def __init__(self, config):
#         super(BertLastFourEmbeddings, self).__init__()
#         self.n_classes = config.num_class
#         config_json = 'bert_config.json' if os.path.exists(config.model_path + 'bert_config.json') else 'config.json'
#         self.bert_config = CONFIGS[config.model].from_pretrained(config.model_path + config_json,
#                                                                  output_hidden_states=True)
#         self.bert_model = MODELS[config.model].from_pretrained(config.model_path, config=self.bert_config)
#         self.isDropout = True if 0 < config.dropout < 1 else False
#         self.dropout = nn.Dropout(p=config.dropout)
#         self.classifier = nn.Linear(self.bert_config.hidden_size * 4, config.num_class)
#
#     def forward(self, input_ids, input_masks, segment_ids):
#         sequence_output, pooler_output, hidden_states = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids,
#                                                                         attention_mask=input_masks)
#
#         hidden_states1 = torch.mean(hidden_states[-1], dim=1)
#         hidden_states2 = torch.mean(hidden_states[-2], dim=1)
#         hidden_states3 = torch.mean(hidden_states[-3], dim=1)
#         hidden_states4 = torch.mean(hidden_states[-4], dim=1)
#         output = torch.cat(
#             (hidden_states1, hidden_states2, hidden_states3, hidden_states4), dim=1)
#         if self.isDropout:
#             output = self.dropout(output)
#         logit = self.classifier(output)
#
#         return logit
#
#
# class BertLastFourEmbeddingsPooler(nn.Module):
#     def __init__(self, config):
#         super(BertLastFourEmbeddingsPooler, self).__init__()
#         self.n_classes = config.num_class
#         config_json = 'bert_config.json' if os.path.exists(config.model_path + 'bert_config.json') else 'config.json'
#         self.bert_config = CONFIGS[config.model].from_pretrained(config.model_path + config_json,
#                                                                  output_hidden_states=True)
#         self.bert_model = MODELS[config.model].from_pretrained(config.model_path, config=self.bert_config)
#         self.isDropout = True if 0 < config.dropout < 1 else False
#         self.dropout = nn.Dropout(p=config.dropout)
#         self.classifier = nn.Linear(self.bert_config.hidden_size * 5, config.num_class)
#
#     def forward(self, input_ids, input_masks, segment_ids):
#         sequence_output, pooler_output, hidden_states = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids,
#                                                                         attention_mask=input_masks)
#
#         hidden_states1 = torch.mean(hidden_states[-1], dim=1)
#         hidden_states2 = torch.mean(hidden_states[-2], dim=1)
#         hidden_states3 = torch.mean(hidden_states[-3], dim=1)
#         hidden_states4 = torch.mean(hidden_states[-4], dim=1)
#         output = torch.cat(
#             (pooler_output, hidden_states1, hidden_states2, hidden_states3, hidden_states4), dim=1)
#         if self.isDropout:
#             output = self.dropout(output)
#         logit = self.classifier(output)
#
#         return logit
#
# class BertDynCls(nn.Module):
#     def __init__(self, config):
#         super(BertDynCls, self).__init__()
#         self.n_classes = config.num_class
#         config_json = 'bert_config.json' if os.path.exists(config.model_path + 'bert_config.json') else 'config.json'
#         self.bert_config = CONFIGS[config.model].from_pretrained(config.model_path + config_json,
#                                                                  output_hidden_states=True)
#         self.bert_model = MODELS[config.model].from_pretrained(config.model_path, config=self.bert_config)
#         self.dynWeight = nn.Linear(self.bert_config.hidden_size, 1)
#         self.dence = nn.Linear(self.bert_config.hidden_size, 512)
#         self.isDropout = True if 0 < config.dropout < 1 else False
#         self.dropout = nn.Dropout(p=config.dropout)
#         self.classifier = nn.Linear(512, config.num_class)
#
#
#     def forward(self, input_ids, input_masks, segment_ids):
#         sequence_output, pooler_output, hidden_states = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids,
#                                                                         attention_mask=input_masks)
#
#         batch_size = pooler_output.shape[0]
#
#         hid_avg_list = None
#         weight_list = None
#         for i, hidden in enumerate(hidden_states):
#             hid_avg = hidden_states[-(i + 1)][0]
#             weight = self.dynWeight(hid_avg).repeat(1, self.bert_config.hidden_size)
#             if hid_avg_list is None:
#                 hid_avg_list = hid_avg
#             else:
#                 hid_avg_list = torch.cat((hid_avg_list, hid_avg), dim=1)
#
#             if weight_list is None:
#                 weight_list = hid_avg
#             else:
#                 weight_list = torch.cat((weight_list, weight), dim=1)
#
#         concat_out = weight_list.mul_(hid_avg_list)
#         concat_out = concat_out.reshape(batch_size, -1, self.bert_config.hidden_size)
#         concat_out = torch.sum(concat_out, dim=1)
#
#         if self.isDropout:
#             concat_out = self.dropout(concat_out)
#         concat_out = self.dence(concat_out)
#         logit = self.classifier(concat_out)
#
#         return logit
#
# class BertDynEmbeddings(nn.Module):
#     def __init__(self, config):
#         super(BertDynEmbeddings, self).__init__()
#         self.n_classes = config.num_class
#         config_json = 'bert_config.json' if os.path.exists(config.model_path + 'bert_config.json') else 'config.json'
#         self.bert_config = CONFIGS[config.model].from_pretrained(config.model_path + config_json,
#                                                                  output_hidden_states=True)
#         self.bert_model = MODELS[config.model].from_pretrained(config.model_path, config=self.bert_config)
#         self.dynWeight = nn.Linear(self.bert_config.hidden_size, 1)
#         self.dence = nn.Linear(self.bert_config.hidden_size, 512)
#         self.isDropout = True if 0 < config.dropout < 1 else False
#         self.dropout = nn.Dropout(p=config.dropout)
#         self.classifier = nn.Linear(512, config.num_class)
#
#
#     def forward(self, input_ids, input_masks, segment_ids):
#         sequence_output, pooler_output, hidden_states = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids,
#                                                                         attention_mask=input_masks)
#
#         batch_size = pooler_output.shape[0]
#
#         hid_avg_list = None
#         weight_list = None
#         for i, hidden in enumerate(hidden_states):
#             hid_avg = torch.mean(hidden_states[-(i + 1)], dim=1)
#             weight = self.dynWeight(hid_avg).repeat(1, self.bert_config.hidden_size)
#             if hid_avg_list is None:
#                 hid_avg_list = hid_avg
#             else:
#                 hid_avg_list = torch.cat((hid_avg_list, hid_avg), dim=1)
#
#             if weight_list is None:
#                 weight_list = hid_avg
#             else:
#                 weight_list = torch.cat((weight_list, weight), dim=1)
#
#         concat_out = weight_list.mul_(hid_avg_list)
#         concat_out = concat_out.reshape(batch_size, -1, self.bert_config.hidden_size)
#         concat_out = torch.sum(concat_out, dim=1)
#
#         if self.isDropout:
#             concat_out = self.dropout(concat_out)
#
#         concat_out = self.dence(concat_out)
#         logit = self.classifier(concat_out)
#
#         return logit
#
#
# class BertRNN(nn.Module):
#
#     def __init__(self, config):
#         super(BertRNN, self).__init__()
#         self.rnn_type = "gru"
#         self.bidirectional = True
#         self.hidden_dim = 256
#         self.n_layers = 2
#         self.batch_first = True
#         self.drop_out = 0.1
#         self.n_classes = config.num_class
#         config_json = 'bert_config.json' if os.path.exists(config.model_path + 'bert_config.json') else 'config.json'
#         self.bert_config = CONFIGS[config.model].from_pretrained(config.model_path + config_json,
#                                                                  output_hidden_states=True)
#         self.bert_model = MODELS[config.model].from_pretrained(config.model_path, config=self.bert_config)
#         self.num_directions = 1 if not self.bidirectional else 2
#
#         if self.rnn_type == 'lstm':
#             self.rnn = nn.LSTM(self.bert_config.to_dict()['hidden_size'],
#                                hidden_size=self.hidden_dim,
#                                num_layers=self.n_layers,
#                                bidirectional=self.bidirectional,
#                                batch_first=self.batch_first,
#                                dropout=self.drop_out)
#         elif self.rnn_type == 'gru':
#             self.rnn = nn.GRU(self.bert_config.to_dict()['hidden_size'],
#                               hidden_size=self.hidden_dim,
#                               num_layers=self.n_layers,
#                               bidirectional=self.bidirectional,
#                               batch_first=self.batch_first,
#                               dropout=self.drop_out)
#         else:
#             self.rnn = nn.RNN(self.bert_config.to_dict()['hidden_size'],
#                               hidden_size=self.hidden_dim,
#                               num_layers=self.n_layers,
#                               bidirectional=self.bidirectional,
#                               batch_first=self.batch_first,
#                               dropout=self.drop_out)
#
#         self.dropout = nn.Dropout(self.drop_out)
#         self.fc_rnn = nn.Linear(self.hidden_dim * self.num_directions, config.num_class)
#
#     def forward(self, input_ids, input_masks, segment_ids):
#
#         sequence_output, pooler_output, hidden_states = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids,
#                                                                         attention_mask=input_masks)
#
#         self.rnn.flatten_parameters()
#         if self.rnn_type in ['rnn', 'gru']:
#             output, hidden = self.rnn(sequence_output)
#         else:
#             output, (hidden, cell) = self.rnn(sequence_output)
#
#         # output = [ batch size, sent len, hidden_dim * bidirectional]
#         batch_size, max_seq_len, hidden_dim = output.shape
#         hidden = torch.transpose(hidden, 1, 0)
#         hidden = torch.mean(torch.reshape(hidden, [batch_size, -1, hidden_dim]), dim=1)
#         output = torch.sum(output, dim=1)
#         fc_input = self.dropout(output + hidden)
#
#         # output = torch.mean(output, dim=1)
#         # fc_input = self.dropout(output)
#         out = self.fc_rnn(fc_input)
#
#         return out
#
#
# class BertCNN(nn.Module):
#
#     def __init__(self, config):
#         super(BertCNN, self).__init__()
#         self.num_filters = 100
#         config_json = 'bert_config.json' if os.path.exists(config.model_path + 'bert_config.json') else 'config.json'
#         self.bert_config = CONFIGS[config.model].from_pretrained(config.model_path + config_json,
#                                                                  output_hidden_states=True)
#         self.hidden_size = self.bert_config.to_dict()['hidden_size']
#         self.filter_sizes = {3, 4, 5}
#         self.drop_out = 0.5
#
#         self.convs = nn.ModuleList(
#             [nn.Conv2d(1, self.num_filters, (k, self.hidden_size)) for k in self.filter_sizes])
#
#
#         self.bert_model = MODELS[config.model].from_pretrained(config.model_path, config=self.bert_config)
#         self.dropout = nn.Dropout(self.drop_out)
#
#         self.fc_cnn = nn.Linear(self.num_filters * len(self.filter_sizes), config.num_class)
#
#     def conv_and_pool(self, x, conv):
#         x = F.relu(conv(x)).squeeze(3)
#         x = F.max_pool1d(x, x.size(2)).squeeze(2)
#         return x
#
#     def forward(self, input_ids, input_masks, segment_ids):
#         sequence_output, pooler_output, hidden_states = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids,
#                                                                         attention_mask=input_masks)
#
#         sequence_output = self.dropout(sequence_output)
#         out = sequence_output.unsqueeze(1)
#         out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
#         out = self.dropout(out)
#         out = self.fc_cnn(out)
#         return out
#
#
# class BertRCNN(nn.Module):
#     def __init__(self, config):
#         super(BertRCNN, self).__init__()
#         self.rnn_type = "lstm"
#         self.bidirectional = True
#         self.hidden_dim = 256
#         self.n_layers = 2
#         self.batch_first = True
#         self.drop_out = 0.5
#
#         config_json = 'bert_config.json' if os.path.exists(config.model_path + 'bert_config.json') else 'config.json'
#         self.bert_config = CONFIGS[config.model].from_pretrained(config.model_path + config_json,
#                                                                  output_hidden_states=True)
#
#         if self.rnn_type == 'lstm':
#             self.rnn = nn.LSTM(self.bert_config.to_dict()['hidden_size'],
#                                self.hidden_dim,
#                                num_layers=self.n_layers,
#                                bidirectional=self.bidirectional,
#                                batch_first=self.batch_first,
#                                dropout=self.drop_out)
#         elif self.rnn_type == 'gru':
#             self.rnn = nn.GRU(self.bert_config.to_dict()['hidden_size'],
#                               hidden_size=self.hidden_dim,
#                               num_layers=self.n_layers,
#                               bidirectional=self.bidirectional,
#                               batch_first=self.batch_first,
#                               dropout=self.drop_out)
#         else:
#             self.rnn = nn.RNN(self.bert_config.to_dict()['hidden_size'],
#                               hidden_size=self.hidden_dim,
#                               num_layers=self.n_layers,
#                               bidirectional=self.bidirectional,
#                               batch_first=self.batch_first,
#                               dropout=self.drop_out)
#
#         # self.maxpool = nn.MaxPool1d()
#
#
#         self.bert_model = MODELS[config.model].from_pretrained(config.model_path, config=self.bert_config)
#         self.fc = nn.Linear(self.hidden_dim * self.n_layers, config.num_class)
#         self.dropout = nn.Dropout(self.drop_out)
#
#     def forward(self, input_ids, input_masks, segment_ids):
#
#         sequence_output, pooler_output, hidden_states = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids,
#                                                                         attention_mask=input_masks)
#
#         sentence_len = sequence_output.shape[1]
#         pooler_output = pooler_output.unsqueeze(dim=1).repeat(1, sentence_len, 1)
#         bert_sentence = sequence_output + pooler_output
#
#         self.rnn.flatten_parameters()
#         if self.rnn_type in ['rnn', 'gru']:
#             output, hidden = self.rnn(bert_sentence)
#         else:
#             output, (hidden, cell) = self.rnn(bert_sentence)
#
#         batch_size, max_seq_len, hidden_dim = output.shape
#         out = torch.transpose(output.relu(), 1, 2)
#
#         out = F.max_pool1d(out, max_seq_len).squeeze()
#         out = self.fc(out)
#
#         return out
#
#
# class XLNet(nn.Module):
#
#     def __init__(self, config):
#         super(XLNet, self).__init__()
#         self.xlnet = XLNetModel.from_pretrained(config.model_path)
#
#         self.isDropout = True if 0 < config.dropout < 1 else False
#         self.dropout = nn.Dropout(p=config.dropout)
#         self.fc = nn.Linear(self.xlnet.d_model, config.num_class)
#
#     def forward(self, input_ids, input_masks, segment_ids):
#         sequence_output = self.xlnet(input_ids=input_ids, token_type_ids=segment_ids,
#                                      attention_mask=input_masks)
#         sequence_output = torch.sum(sequence_output[0], dim=1)
#         if self.isDropout:
#             sequence_output = self.dropout(sequence_output)
#         out = self.fc(sequence_output)
#         return out
#
#
# class ElectraClassificationHead(nn.Module):
#     """Head for sentence-level classification tasks."""
#
#     def __init__(self, config):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
#
#     def forward(self, features, **kwargs):
#         x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
#         x = self.dropout(x)
#         x = self.dense(x)
#         x = get_activation("gelu")(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
#         x = self.dropout(x)
#         x = self.out_proj(x)
#         return x
#
# class Electra(nn.Module):
#
#     def __init__(self, config):
#         super(Electra, self).__init__()
#         self.electra = ElectraModel.from_pretrained(config.model_path)
#
#         config_json = 'bert_config.json' if os.path.exists(config.model_path + 'bert_config.json') else 'config.json'
#         self.electra_config = CONFIGS[config.model].from_pretrained(config.model_path + config_json)
#         self.electra_config.num_labels = config.num_class
#         self.fc = ElectraClassificationHead(self.electra_config)
#
#     def forward(self, input_ids, input_masks, segment_ids):
#         discriminator_hidden_states = self.electra(input_ids=input_ids, token_type_ids=segment_ids,
#                                      attention_mask=input_masks)
#
#         sequence_output = discriminator_hidden_states[0]
#         out = self.fc(sequence_output)
#         return out
#
# class NEZHA(nn.Module):
#     def __init__(self, config):
#         super(NEZHA, self).__init__()
#         self.n_classes = config.num_class
#
#         config_json = 'bert_config.json' if os.path.exists(config.model_path + 'bert_config.json') else 'config.json'
#         self.bert_config = CONFIGS[config.model].from_pretrained(config.model_path + config_json)
#         #self.bert_model = MODELS[config.model](config=self.bert_config)
#         self.bert_model = MODELS[config.model].from_pretrained(config.model_path, config=self.bert_config)
#
#         # NEZHA init
#         #torch_init_model(self.bert_model, os.path.join(config.model_path, 'pytorch_model.bin'))
#         self.isDropout = True if 0 < config.dropout < 1 else False
#         self.dropout = nn.Dropout(p=config.dropout)
#         self.classifier = nn.Linear(self.bert_config.hidden_size * 2, self.n_classes)
#
#     def forward(self, input_ids, input_masks, segment_ids):
#         sequence_output, pooler_output = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids,
#                                                                         attention_mask=input_masks)
#         seq_avg = torch.mean(sequence_output, dim=1)
#         concat_out = torch.cat((seq_avg, pooler_output), dim=1)
#
#         if self.isDropout:
#             concat_out = self.dropout(concat_out)
#         logit = self.classifier(concat_out)
#         return logit
#

