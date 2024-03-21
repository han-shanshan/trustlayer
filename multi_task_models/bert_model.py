"""
Implementation borrowed from transformers package and extended to support multiple prediction heads:

https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_bert.py
"""

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import transformers
from transformers import BertTokenizer
from transformers import models
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BERT_INPUTS_DOCSTRING,
    # _TOKENIZER_FOR_DOC,
    _CHECKPOINT_FOR_DOC,
    _CONFIG_FOR_DOC,
    BertModel,
)

from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings_to_model_forward,
)

from transformers import Trainer
from constants import TOPIC_TASK_NAME, SEMANTIC_TASK_NAME



class BertMultiHeadClassifier(nn.Module):
    def __init__(self, bert_model_name, label_dict, id2label, label2id):
        super(BertMultiHeadClassifier, self).__init__()
        
        # Load pre-trained BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # Define two separate classification heads
        self.classifier_topic = nn.Linear(self.bert.config.hidden_size, len(label_dict[TOPIC_TASK_NAME]))
        self.classifier_semantic = nn.Linear(self.bert.config.hidden_size, len(label_dict[SEMANTIC_TASK_NAME]))
        self.label_dict = label_dict
        self.id2label = id2label
        self.label2id = label2id

    def forward(self, input_ids, attention_mask, token_type_ids, task_name=-1):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        if task_name == TOPIC_TASK_NAME:
            return self.classifier_topic(pooled_output)
        elif task_name == SEMANTIC_TASK_NAME:
            return self.classifier_semantic(pooled_output)
        logits_task_topic = self.classifier_topic(outputs.pooler_output)
        logits_task_semantic = self.classifier_semantic(outputs.pooler_output)
        loss = None
        # if TOPIC_TASK_NAME in self.label_dict and SEMANTIC_TASK_NAME in self.label_dict:
        #     loss_fct = nn.BCEWithLogitsLoss()
        #     print(f"logits_task_topic.view(-1, self.classifier_topic.out_features) = {logits_task_topic.view(-1, self.classifier_topic.out_features)}")
        #     print(f"shape of logits_task_topic.view(-1, self.classifier_topic.out_features) = {logits_task_topic.view(-1, self.classifier_topic.out_features).shape}")
        #     # print(f"------torch.tensor(self.label_dict[TOPIC_TASK_NAME]).view(-1) = {torch.tensor(self.id2label[TOPIC_TASK_NAME].keys()).view(-1)}")
        #     loss_task_a = loss_fct(logits_task_topic.view(-1, self.classifier_topic.out_features), torch.tensor(self.id2label[TOPIC_TASK_NAME].keys()).view(-1))
        #     loss_task_b = loss_fct(logits_task_semantic.view(-1, self.classifier_semantic.out_features), torch.tensor(self.label_dict[SEMANTIC_TASK_NAME]).view(-1))
        #     loss = loss_task_a + loss_task_b

        return loss, logits_task_topic, logits_task_semantic


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(transformers.PretrainedConfig())
        self.num_labels = kwargs.get("task_labels_map", {})
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        ## add task specific output heads
        print(f"=========list(self.num_labels.values())[0] = {list(self.num_labels.values())[0]}")
        self.classifier1 = nn.Linear(
            config.hidden_size, list(self.num_labels.values())[0]
        )
        # self.classifier2 = nn.Linear(
        #     config.hidden_size, list(self.num_labels.values())[1]
        # )

        self.init_weights()

    @add_start_docstrings_to_model_forward(
        BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        # tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_name=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = None
        if task_name == list(self.num_labels.keys())[0]:
            logits = self.classifier1(pooled_output)
        # elif task_name == list(self.num_labels.keys())[1]:
        #     logits = self.classifier2(pooled_output)
        print(f"task name = {task_name}")
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels[task_name] == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels[task_name] > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels[task_name] == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels[task_name]), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
