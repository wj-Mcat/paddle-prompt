"""mixed template"""
# %%
from collections import OrderedDict
from paddle import nn
from paddle.optimizer import AdamW

from paddlenlp.transformers.t5.modeling import T5ForConditionalGeneration
from paddlenlp.transformers.t5.tokenizer import T5Tokenizer
from paddlenlp.datasets import load_dataset

from paddle_prompt.config import Config
from paddle_prompt.templates.mixed_template import MixedTemplate
from paddle_prompt.verbalizers.manual_verbalizer import ManualVerbalizer
from paddle_prompt.schema import InputExample

## 1. prepare the dataset
train, dev = load_dataset(
    'glue', 'sst-2',
    splits=['train', 'dev']
)
train_examples = list(train.map(lambda x: InputExample(text_a=x['sentence'], label=x['labels'])))
dev_examples = list(dev.map(lambda x: InputExample(text_a=x['sentence'], label=x['labels'])))

## 2. prepare the prompt model related
config: Config = Config().parse_args(known_only=True)
config.pretrained_model = 't5-small'

tokenizer = T5Tokenizer.from_pretrained(config.pretrained_model)
plm = T5ForConditionalGeneration.from_pretrained(config.pretrained_model)

label2words = OrderedDict({'0': 'negative', '1': 'positive'})
template = MixedTemplate(
    tokenizer=tokenizer,
    plm=plm,
    config=config,
    label2words=label2words,
)

verbalizer = ManualVerbalizer(tokenizer=tokenizer, label2words=label2words, config=config)

# %%

## 3. prepare the training related
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(parameters=template.parameters(), learning_rate=config.learning_rate)

# wrapped_example = mytemplate.wrap_one_example(dataset['train'][0]) 
# print(wrapped_example)

# # %%

# wrapped_t5tokenizer = WrapperClass(max_seq_length=128, decoder_max_length=3, tokenizer=tokenizer, truncate_method="head")

# from openprompt import PromptDataLoader

# train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer, 
#     tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3, 
#     batch_size=4,shuffle=True, teacher_forcing=False, predict_eos_token=False,
#     truncate_method="head")
# # next(iter(train_dataloader))

# # ## Define the verbalizer
# # In classification, you need to define your verbalizer, which is a mapping from logits on the vocabulary to the final label probability. Let's have a look at the verbalizer details:

# # %%

# from openprompt.prompts import ManualVerbalizer
# import torch

# # for example the verbalizer contains multiple label words in each class
# myverbalizer = ManualVerbalizer(tokenizer, num_classes=2, 
#                         label_words=[["yes"], ["no"], ["maybe"]])

# print(myverbalizer.label_words_ids)
# logits = torch.randn(2,len(tokenizer)) # creating a pseudo output from the plm
# a = myverbalizer.process_logits(logits)

# from openprompt import PromptForClassification

# use_cuda = True
# prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
# if use_cuda:
#     prompt_model=  prompt_model.cuda()

# # ## below is standard training


# from transformers import  AdamW, get_linear_schedule_with_warmup
# loss_func = torch.nn.CrossEntropyLoss()

# no_decay = ['bias', 'LayerNorm.weight']

# # it's always good practice to set no decay to biase and LayerNorm parameters
# optimizer_grouped_parameters1 = [
#     {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
#     {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
# ]

# # Using different optimizer for prompt parameters and model parameters
# optimizer_grouped_parameters2 = [
#     {'params': [p for n,p in prompt_model.template.named_parameters() if "raw_embedding" not in n]}
# ]

# optimizer1 = AdamW(optimizer_grouped_parameters1, lr=1e-4)
# optimizer2 = AdamW(optimizer_grouped_parameters2, lr=1e-3)

# for epoch in range(10):
#     tot_loss = 0 
#     for step, inputs in enumerate(train_dataloader):
#         if use_cuda:
#             inputs = inputs.cuda()
#         logits = prompt_model(inputs)
#         labels = inputs['label']
#         loss = loss_func(logits, labels)
#         loss.backward()
#         tot_loss += loss.item()
#         optimizer1.step()
#         optimizer1.zero_grad()
#         optimizer2.step()
#         optimizer2.zero_grad()
#         print(tot_loss/(step+1))
    
# # ## evaluate

# # %%

# # 在预测的时候，是没有办法指定哪种template，在这里只用了一个template
# validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer, 
#     tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=3, 
#     batch_size=4,shuffle=False, teacher_forcing=False, predict_eos_token=False,
#     truncate_method="head")


# allpreds = []
# alllabels = []
# for step, inputs in enumerate(validation_dataloader):
#     if use_cuda:
#         inputs = inputs.cuda()
#     logits = prompt_model(inputs)
#     labels = inputs['label']
#     alllabels.extend(labels.cpu().tolist())
#     allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

# acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
# print(acc)
