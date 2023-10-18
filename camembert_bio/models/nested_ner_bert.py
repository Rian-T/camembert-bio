import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel
from seqeval.metrics import f1_score, precision_score, recall_score
from seqeval.scheme import IOB2


class NestedPerClassNERModel(pl.LightningModule):
    def __init__(
        self, n_classes, n_layers, id2label, pretrained_model_name="camembert-base"
    ):
        super(NestedPerClassNERModel, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.ner_layers = nn.ModuleList(
            [
                nn.Linear(self.bert.config.hidden_size, n_classes)
                for _ in range(n_layers)
            ]
        )
        self.n_layers = n_layers
        self.id2label = id2label

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs["last_hidden_state"]
        all_logits = [ner_layer(sequence_output) for ner_layer in self.ner_layers]
        return all_logits

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
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5)
        return optimizer


class NestedPerDepthNERModel(pl.LightningModule):
    def __init__(self, n_depth, id2label, pretrained_model_name="camembert-base", learning_rate=5e-5, dropout_prob=0.1, stack_depths=False):
        super(NestedPerDepthNERModel, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.id2label = id2label
        self.learning_rate = learning_rate
        self.stack_depths = stack_depths

        # One common dense layer for transformation, not for prediction
        self.common_layer = nn.Linear(
            self.bert.config.hidden_size, self.bert.config.hidden_size
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = nn.ReLU()

        # One dense layer per depth for prediction
        self.depth_layers = nn.ModuleList(
            [
                nn.Linear(self.bert.config.hidden_size, len(id2label))
                for _ in range(n_depth)
            ]
        )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]

        # Pass through the common transformation layer
        transformed_output = self.common_layer(sequence_output)
        transformed_output = self.activation(transformed_output)
        transformed_output = self.dropout(transformed_output)

        depth_logits = []
        if self.stack_depths:
            for layer in self.depth_layers:
                transformed_output = layer(transformed_output)
                depth_logits.append(transformed_output)
                transformed_output = self.activation(transformed_output)  # Optional: You can add activation and dropout here too if needed.
                transformed_output = self.dropout(transformed_output)
        else:
            depth_logits = [layer(transformed_output) for layer in self.depth_layers]

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
