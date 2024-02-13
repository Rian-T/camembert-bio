import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer, AutoModelForTokenClassification
from seqeval.metrics import f1_score, precision_score, recall_score
from seqeval.scheme import IOB2
import stanza

from camembert_bio.utils.offsets_evaluation import tags_to_entities_with_offsets

class NERModel(pl.LightningModule):
    def __init__(self, id2label, pretrained_model_name="camembert-base", learning_rate=5e-5, dropout_prob=0.1, train_head_only=False):
        super(NERModel, self).__init__()
        
        self.id2label = id2label
        self.learning_rate = learning_rate
        self.nlp = stanza.Pipeline(lang='fr', processors='tokenize')
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.train_head_only = train_head_only

        self.bert = AutoModelForTokenClassification.from_pretrained(pretrained_model_name, num_labels=len(id2label))

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
        return F.cross_entropy(logits.view(-1, len(self.id2label)), labels.view(-1), ignore_index=-100)

    def logits_to_tags(self, logits):
        """Convert logits to tag sequences."""
        tag_ids = torch.argmax(logits, dim=-1).cpu().numpy()
        return [[self.id2label[tag_id] for tag_id in sequence] for sequence in tag_ids]

    def labels_to_tags(self, labels):
        """Convert label IDs to tag sequences, skipping -100 values."""
        return [
            [self.id2label[label_id] if label_id != -100 else "O" for label_id in sequence]
            for sequence in labels.cpu().numpy()
        ]

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        loss = self.compute_loss(logits, labels)

        # Compute metrics
        preds = self.logits_to_tags(logits)
        labels = self.labels_to_tags(labels)

        self.log("train/loss", loss)
        self.log("train/precision", precision_score(labels, preds))
        self.log("train/recall", recall_score(labels, preds))
        self.log("train/f1", f1_score(labels, preds))

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        val_loss = self.compute_loss(logits, labels)

        # Compute metrics
        preds = self.logits_to_tags(logits)
        labels = self.labels_to_tags(labels)

        self.log("val/loss", val_loss)
        self.log("val/precision", precision_score(labels, preds))
        self.log("val/recall", recall_score(labels, preds))
        self.log("val/f1", f1_score(labels, preds))

        return val_loss
    
    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        test_loss = self.compute_loss(logits, labels)

        # Compute metrics
        preds = self.logits_to_tags(logits)
        labels = self.labels_to_tags(labels)

        self.log("test/loss", test_loss)
        self.log("test/precision", precision_score(labels, preds))
        self.log("test/recall", recall_score(labels, preds))
        self.log("test/f1", f1_score(labels, preds))

        return test_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    def predict_example(self, example):
        text = example['passages'][0]['text'][0]
        doc = self.nlp(text)
        predicted_entities = []

        for sentence in doc.sentences:
            tokens = [token.text for token in sentence.tokens]
            encoding = self.tokenizer(
                tokens, 
                is_split_into_words=True, 
                truncation=True, 
                padding='max_length', 
                max_length=512, 
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].squeeze()
            attention_mask = encoding['attention_mask'].squeeze()

            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

            logits = self(input_ids, attention_mask)
            tags = self.logits_to_tags(logits)[0]  # Assuming single sentence

            sentence_predicted_entities = tags_to_entities_with_offsets(tokens, tags, sentence.text)
            predicted_entities.extend(sentence_predicted_entities)

        return predicted_entities
    
    def predict_batch(self, examples):
        batch_texts = [passage[0]['text'][0] for passage in examples["passages"]]
        batch_docs = [self.nlp(text) for text in batch_texts]

        # Flatten all sentences from all docs into a single list
        all_sentences = [sentence for doc in batch_docs for sentence in doc.sentences]
        all_tokens = [[token.text for token in sentence.tokens] for sentence in all_sentences]

        # Encode all sentences at once
        encodings = self.tokenizer(
            all_tokens, 
            is_split_into_words=True, 
            truncation=True, 
            padding='max_length', 
            max_length=512, 
            return_tensors='pt'
        ).to(self.device)

        # Get model predictions for all sentences at once
        logits = self(encodings['input_ids'], encodings['attention_mask'])
        all_tags = self.logits_to_tags(logits)

        # Split the predictions back into batches
        batch_predicted_entities = []
        start_index = 0
        for doc in batch_docs:
            predicted_entities = []
            for sentence in doc.sentences:
                tags = all_tags[start_index]
                sentence_predicted_entities = tags_to_entities_with_offsets(all_tokens[start_index], tags, sentence.text)
                predicted_entities.extend(sentence_predicted_entities)
                start_index += 1
            batch_predicted_entities.append(predicted_entities)

        return batch_predicted_entities