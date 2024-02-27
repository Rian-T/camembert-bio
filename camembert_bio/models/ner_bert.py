import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer, AutoModelForTokenClassification
import evaluate
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
import stanza

from camembert_bio.utils.offsets_evaluation import tags_to_entities_with_offsets


class NERModel(pl.LightningModule):
    def __init__(
        self,
        id2label,
        pretrained_model_name="camembert-base",
        learning_rate=5e-5,
        dropout_prob=0.1,
        train_head_only=False,
    ):
        super(NERModel, self).__init__()

        self.id2label = id2label
        self.learning_rate = learning_rate
        # self.nlp = stanza.Pipeline(lang="fr", processors="tokenize")
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.train_head_only = train_head_only

        self.seqeval = evaluate.load("seqeval")

        self.train_step_outputs = []
        self.test_step_outputs = []

        self.bert = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name, num_labels=len(id2label)
        )

        if train_head_only:
            # Freeze BERT or RoBERTa layers
            if hasattr(self, "roberta"):
                for param in self.roberta.parameters():
                    param.requires_grad = False
            else:
                for param in self.bert.parameters():
                    param.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        return logits

    def compute_loss(self, logits, labels):
        return F.cross_entropy(
            logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100
        )

    def logits_to_tags(self, logits):
        """Convert logits to tag sequences."""
        tag_ids = torch.argmax(logits, dim=-1).cpu().numpy()
        return [[self.id2label[tag_id] for tag_id in sequence] for sequence in tag_ids]

    def labels_to_tags(self, labels):
        """Convert label IDs to tag sequences, skipping -100 values."""
        return [
            [
                self.id2label[label_id] if label_id != -100 else "O"
                for label_id in sequence
            ]
            for sequence in labels.cpu().numpy()
        ]

    def compute_metrics(self, preds, labels):
        # Convert logits to predictions
        predictions = torch.argmax(preds, dim=2).cpu().numpy()
        labels = labels.cpu().numpy()

        true_predictions = [
            [self.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = classification_report(
            true_labels,
            true_predictions,
            output_dict=True,
            zero_division=0,
            mode="strict",
            scheme=IOB2,
        )

        def compute_average_metrics(avg):
            return {
                f"{avg}_precision": results[f"{avg} avg"]["precision"],
                f"{avg}_recall": results[f"{avg} avg"]["recall"],
                f"{avg}_f1": results[f"{avg} avg"]["f1-score"],
            }

        return {
            **compute_average_metrics("micro"),
            **compute_average_metrics("macro"),
            **compute_average_metrics("weighted"),
        }

    def common_step(self, batch, batch_idx, prefix):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        loss = self.compute_loss(logits, labels)

        # Compute metrics
        metrics = self.compute_metrics(logits, labels)

        self.log(
            f"{prefix}/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log_dict({f"{prefix}/{k}": v for k, v in metrics.items()})

        return loss, logits, labels

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.common_step(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self.common_step(batch, batch_idx, "val")
        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, labels = self.common_step(batch, batch_idx, "test")
        self.test_step_outputs.append({"predictions": logits, "labels": labels})
        return {"predictions": logits, "labels": labels, "loss": loss}

    def on_test_epoch_end(self):
        # Concatenate all the predictions and labels
        all_predictions = torch.cat(
            [x["predictions"] for x in self.test_step_outputs], dim=0
        )
        all_labels = torch.cat([x["labels"] for x in self.test_step_outputs], dim=0)

        # Compute the loss for the whole dataset
        test_loss = self.compute_loss(all_predictions, all_labels)

        # Compute the metrics for the whole dataset
        metrics = self.compute_metrics(all_predictions, all_labels)

        self.log("test/loss", test_loss)
        self.log_dict({f"test/{k}": v for k, v in metrics.items()})

        self.test_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def predict_example(self, example):
        text = example["passages"][0]["text"][0]
        doc = self.nlp(text)
        predicted_entities = []

        for sentence in doc.sentences:
            tokens = [token.text for token in sentence.tokens]
            encoding = self.tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].squeeze()
            attention_mask = encoding["attention_mask"].squeeze()

            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

            logits = self(input_ids, attention_mask)
            tags = self.logits_to_tags(logits)[0]  # Assuming single sentence

            sentence_predicted_entities = tags_to_entities_with_offsets(
                tokens, tags, sentence.text
            )
            predicted_entities.extend(sentence_predicted_entities)

        return predicted_entities

    def predict_batch(self, examples):
        batch_texts = [passage[0]["text"][0] for passage in examples["passages"]]
        batch_docs = [self.nlp(text) for text in batch_texts]

        # Flatten all sentences from all docs into a single list
        all_sentences = [sentence for doc in batch_docs for sentence in doc.sentences]
        all_tokens = [
            [token.text for token in sentence.tokens] for sentence in all_sentences
        ]

        # Encode all sentences at once
        encodings = self.tokenizer(
            all_tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        # Get model predictions for all sentences at once
        logits = self(encodings["input_ids"], encodings["attention_mask"])
        all_tags = self.logits_to_tags(logits)

        # Split the predictions back into batches
        batch_predicted_entities = []
        start_index = 0
        for doc in batch_docs:
            predicted_entities = []
            for sentence in doc.sentences:
                tags = all_tags[start_index]
                sentence_predicted_entities = tags_to_entities_with_offsets(
                    all_tokens[start_index], tags, sentence.text
                )
                predicted_entities.extend(sentence_predicted_entities)
                start_index += 1
            batch_predicted_entities.append(predicted_entities)

        return batch_predicted_entities
