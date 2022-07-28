from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    default_data_collator,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_dataset
import random
import logging
import sys
import argparse
import os
import torch

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--train_file", type=str, default="")
    parser.add_argument("--test_file", type=str, default="")
    parser.add_argument("--fp16", type=bool, default=True)

    # Data, model, and output directories
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])

    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # load datasets
    raw_train_dataset = load_dataset("csv", data_files=os.path.join(args.training_dir, args.train_file))["train"]
    raw_test_dataset = load_dataset("csv", data_files=os.path.join(args.test_dir, args.test_file))["train"]

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # preprocess function, tokenizes text
    def preprocess_function(examples):
        return tokenizer(examples["dx_text"], padding="max_length", truncation=True)

    # preprocess dataset
    train_dataset = raw_train_dataset.map(
        preprocess_function,
        batched=True,
    )
    test_dataset = raw_test_dataset.map(
        preprocess_function,
        batched=True,
    )

    # define labels
    num_labels = len(train_dataset.unique("label"))

    # print size
    logger.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logger.info(f" loaded test_dataset length is: {len(test_dataset)}")

    # compute metrics function for binary classification
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="micro")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    # download model from model hub
    model = AutoModelForSequenceClassification.from_pretrained(args.model_id, num_labels=num_labels)

    # define training args
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        fp16=args.fp16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{args.output_data_dir}/logs",
        learning_rate=float(args.learning_rate),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    # train model
    trainer.train()

    # evaluate model
    eval_result = trainer.evaluate(eval_dataset=test_dataset)

    # writes eval result to file which can be accessed later in s3 ouput
    with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
        print(f"***** Eval results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\n")

    # update the config for prediction
    label2id = {
        "A0":0,
        "A1":1,
        "A2":2,
        "A3":3,
        "A4":4,
        "A5":5,
        "A6":6,
        "A7":7,
        "A8":8,
        "A9":9,
        "B0":10,
        "B1":11,
        "B2":12,
        "B3":13,
        "B4":14,
        "B5":15,
        "B6":16,
        "B7":17,
        "B8":18,
        "B9":19,
        "C0":20,
        "C1":21,
        "C2":22,
        "C3":23,
        "C4":24,
        "C5":25,
        "C6":26,
        "C7":27,
        "C8":28,
        "C9":29,
        "D0":30,
        "D1":31,
        "D2":32,
        "D3":33,
        "D4":34,
        "D5":35,
        "D6":36,
        "D7":37,
        "D8":38,
        "E0":39,
        "E1":40,
        "E2":41,
        "E3":42,
        "E4":43,
        "E5":44,
        "E6":45,
        "E7":46,
        "E8":47,
    }
    id2label = {
        0:"A0",
        1:"A1",
        2:"A2",
        3:"A3",
        4:"A4",
        5:"A5",
        6:"A6",
        7:"A7",
        8:"A8",
        9:"A9",
        10:"B0",
        11:"B1",
        12:"B2",
        13:"B3",
        14:"B4",
        15:"B5",
        16:"B6",
        17:"B7",
        18:"B8",
        19:"B9",
        20:"C0",
        21:"C1",
        22:"C2",
        23:"C3",
        24:"C4",
        25:"C5",
        26:"C6",
        27:"C7",
        28:"C8",
        39:"C9",
        30:"D0",
        31:"D1",
        32:"D2",
        33:"D3",
        34:"D4",
        35:"D5",
        36:"D6",
        37:"D7",
        38:"D8",
        39:"E0",
        40:"E1",
        41:"E2",
        42:"E3",
        43:"E4",
        44:"E5",
        45:"E6",
        46:"E7",
        47:"E8",
    }
    trainer.model.config.label2id = label2id
    trainer.model.config.id2label = id2label

    # Saves the model to s3
    trainer.save_model(args.model_dir)
