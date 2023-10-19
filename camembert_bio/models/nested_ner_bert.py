import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer
from seqeval.metrics import f1_score, precision_score, recall_score
from seqeval.scheme import IOB2
import stanza

from camembert_bio.utils.offsets_evaluation import tags_to_entities_with_offsets

class NestedPerClassNERModel(pl.LightningModule):
    def __init__(self, id2label, pretrained_model_name="camembert-base", learning_rate=5e-5, dropout_prob=0.1, stack_classes=False):
        super(NestedPerClassNERModel, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.id2label = id2label
        self.learning_rate = learning_rate
        self.stack_classes = stack_classes
        self.n_classes = len(id2label.keys())

        # Common transformation layer
        self.common_layer = nn.Linear(
            self.bert.config.hidden_size, self.bert.config.hidden_size
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = nn.ReLU()

        # One prediction layer per class
        self.prediction_layers = nn.ModuleList([nn.Linear(self.bert.config.hidden_size, 3) for _ in range(self.n_classes)])  # 3 for IOB tagging
        self.representation_layers = nn.ModuleList([nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size) for _ in range(self.n_classes)]) if stack_classes else None

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]

        # Pass through the common transformation layer
        transformed_output = self.common_layer(sequence_output)
        transformed_output = self.activation(transformed_output)
        transformed_output = self.dropout(transformed_output)

        class_logits = []
        if self.stack_classes:
            previous_representations = [transformed_output]
            for prediction_layer, representation_layer in zip(self.prediction_layers, self.representation_layers):
                # Average the representations
                avg_representation = torch.mean(torch.stack(previous_representations, dim=0), dim=0)
                # Obtain new representation for the current class
                new_representation = representation_layer(avg_representation)
                # Predict the labels for the current class
                current_logits = prediction_layer(new_representation)
                # Update the list of representations
                previous_representations.append(new_representation)
                # Store the logits
                class_logits.append(current_logits)
        else:
            for prediction_layer in self.prediction_layers:
                current_logits = prediction_layer(transformed_output)
                class_logits.append(current_logits)

        return class_logits

    def compute_loss(self, logits, labels):
        return sum(
            F.cross_entropy(
                logit.view(-1, logit.size(-1)), label.view(-1), ignore_index=-100
            )
            for logit, label in zip(logits, labels)
        )

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, *labels = batch
        logits = self(input_ids, attention_mask)
        loss = self.compute_loss(logits, labels)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def logits_to_tags(self, logits, id2label):
        """Convert logits to tag sequences."""
        tag_ids = torch.argmax(logits, dim=-1).cpu().numpy()
        return [[id2label[tag_id] for tag_id in sequence] for sequence in tag_ids]

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, *labels = batch
        logits = self(input_ids, attention_mask)
        val_loss = self.compute_loss(logits, labels)
        self.log("val_loss", val_loss, prog_bar=True, on_step=True, on_epoch=True)

        # Compute metrics
        all_preds = [self.logits_to_tags(logit, self.id2label) for logit in logits]
        all_labels = [label.cpu().numpy().tolist() for label in labels]

        for depth, (preds, labels) in enumerate(zip(all_preds, all_labels)):
            self.log(
                f"depth_{depth}/val_precision",
                precision_score(labels, preds),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                f"depth_{depth}/val_recall",
                recall_score(labels, preds),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                f"depth_{depth}/val_f1",
                f1_score(labels, preds),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )

        # Global metrics
        global_preds = [tag for sequence in all_preds for tag in sequence]
        global_labels = [tag for sequence in all_labels for tag in sequence]
        self.log(
            "global/val_precision",
            precision_score(global_labels, global_preds),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "global/val_recall",
            recall_score(global_labels, global_preds),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "global/val_f1",
            f1_score(global_labels, global_preds),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


class NestedPerDepthNERModel(pl.LightningModule):
    def __init__(self, n_depth, id2label, pretrained_model_name="camembert-base", learning_rate=5e-5, dropout_prob=0.1, stack_depths=False):
        super(NestedPerDepthNERModel, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.id2label = id2label
        self.learning_rate = learning_rate
        self.stack_depths = stack_depths
        self.nlp = stanza.Pipeline(lang='fr', processors='tokenize')
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        # One common dense layer for transformation, not for prediction
        self.common_layer = nn.Linear(
            self.bert.config.hidden_size, self.bert.config.hidden_size
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = nn.ReLU()

        self.prediction_layers = nn.ModuleList([nn.Linear(self.bert.config.hidden_size, len(id2label)) for _ in range(n_depth)])
        self.representation_layers = nn.ModuleList([nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size) for _ in range(n_depth)]) if stack_depths else None

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]

        # Pass through the common transformation layer
        transformed_output = self.common_layer(sequence_output)
        transformed_output = self.activation(transformed_output)
        transformed_output = self.dropout(transformed_output)

        depth_logits = []
        if self.stack_depths:
            previous_representations = [transformed_output]
            for prediction_layer, representation_layer in zip(self.prediction_layers, self.representation_layers):
                # Average the representations
                avg_representation = torch.mean(torch.stack(previous_representations, dim=0), dim=0)
                # Obtain new representation for the current depth
                new_representation = representation_layer(avg_representation)
                # Predict the labels for the current depth
                current_logits = prediction_layer(new_representation)
                # Update the list of representations
                previous_representations.append(new_representation)
                # Store the logits
                depth_logits.append(current_logits)
        else:
            for prediction_layer in self.prediction_layers:
                current_logits = prediction_layer(transformed_output)
                depth_logits.append(current_logits)

        return depth_logits

    def compute_loss(self, logits, labels):
        return sum(
            F.cross_entropy(
                logit.view(-1, logit.size(-1)), label.view(-1), ignore_index=-100
            )
            for logit, label in zip(logits, labels)
        )
    
    def logits_to_tags(self, logits, id2label):
        """Convert logits to tag sequences."""
        tag_ids = torch.argmax(logits, dim=-1).cpu().numpy()
        return [[id2label[tag_id] for tag_id in sequence] for sequence in tag_ids]
    
    def labels_to_tags(self, labels):
        """Convert label IDs to tag sequences, skipping -100 values."""
        return [
            [self.id2label[label_id] if label_id != -100 else "O" for label_id in sequence]
            for sequence in labels.cpu().numpy()
        ]
    
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, *labels = batch
        logits = self(input_ids, attention_mask)
        loss = self.compute_loss(logits, labels)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        # Compute metrics
        all_preds = [self.logits_to_tags(logit, self.id2label) for logit in logits]
        all_labels = [self.labels_to_tags(label) for label in labels]

        for depth, (preds, labels) in enumerate(zip(all_preds, all_labels)):
            self.log(
                f"depth_{depth}/train_precision",
                float(precision_score(labels, preds)),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                f"depth_{depth}/train_recall",
                float(recall_score(labels, preds)),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                f"depth_{depth}/train_f1",
                float(f1_score(labels, preds)),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )

        # Global metrics
        global_preds = [tag for sequence in all_preds for tag in sequence]
        global_labels = [tag for sequence in all_labels for tag in sequence]
        self.log(
            "global/train_precision",
            float(precision_score(global_labels, global_preds)),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "global/train_recall",
            float(recall_score(global_labels, global_preds)),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "global/train_f1",
            float(f1_score(global_labels, global_preds)),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, *labels = batch
        logits = self(input_ids, attention_mask)
        val_loss = self.compute_loss(logits, labels)
        self.log("val_loss", val_loss, prog_bar=True, on_step=True, on_epoch=True)

        # Compute metrics
        all_preds = [self.logits_to_tags(logit, self.id2label) for logit in logits]
        all_labels = [self.labels_to_tags(label) for label in labels]

        for depth, (preds, labels) in enumerate(zip(all_preds, all_labels)):
            self.log(
                f"depth_{depth}/val_precision",
                float(precision_score(labels, preds)),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                f"depth_{depth}/val_recall",
                float(recall_score(labels, preds)),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                f"depth_{depth}/val_f1",
                float(f1_score(labels, preds)),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )

        # Global metrics
        global_preds = [tag for sequence in all_preds for tag in sequence]
        global_labels = [tag for sequence in all_labels for tag in sequence]
        self.log(
            "global/val_precision",
            float(precision_score(global_labels, global_preds)),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "global/val_recall",
            float(recall_score(global_labels, global_preds)),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "global/val_f1",
            float(f1_score(global_labels, global_preds)),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def predict_example(self, example):
        text = example['passages'][0]['text'][0]  # Get the text from the example
        doc = self.nlp(text)  # Process text with stanza
        
        predicted_entities = []
        
        # Iterate over sentences
        for idx, sentence in enumerate(doc.sentences):
            tokens = [token.text for token in sentence.tokens]  # Tokenize sentence
            offsets = [(token.start_char, token.end_char) for token in sentence.tokens]  # Get token offsets
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)  # Convert tokens to IDs
            input_ids = torch.tensor([input_ids]).to(self.device)  # Assume the model is on the same device
            
            # Get model predictions
            logits_per_depth = self(input_ids)  # This will return a list of logits, one for each depth
            
            # Convert logits to tag sequences and extract entities for each depth
            for logits in logits_per_depth:
                tags = self.logits_to_tags(logits, self.id2label)[0]  # Assuming logits_to_tags returns a list of lists
                sentence_predicted_entities = tags_to_entities_with_offsets(tokens, tags, sentence.text)
                
                # Adjust the offsets based on the position of the sentence in the original text
                sentence_start_offset = text.index(sentence.text)
                for entity in sentence_predicted_entities:
                    entity['offsets'] = [(start + sentence_start_offset, end + sentence_start_offset) for start, end in entity['offsets']]
                
                predicted_entities.extend(sentence_predicted_entities)
        
        return predicted_entities

