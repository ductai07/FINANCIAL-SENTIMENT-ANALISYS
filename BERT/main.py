from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import DataLoader
import torch
from setiment_dataset import create_dataset
from data import df_train, df_val, df_test
import numpy as np
import evaluate
from transformers import AutoConfig

MAX_LEN = 160


model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)


train_dataset = create_dataset(df_train, tokenizer, MAX_LEN )
val_dataset =create_dataset(df_val, tokenizer, MAX_LEN )
test_dataset = create_dataset(df_test, tokenizer, MAX_LEN )


metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    result = metric.compute(predictions=predictions, references=labels)
    return result


num_labels = 3
 
config = AutoConfig.from_pretrained(
    model_name,
    num_labels=num_labels,
    finetuning_task="text-classification"
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    config=config
)

training_args = TrainingArguments(
    output_dir="save_model",
    learning_rate=2e-5,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    num_train_epochs=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)
trainer.train()
trainer.evaluate(test_dataset)