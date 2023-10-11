import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import CamembertModel

class NestedNERModel(pl.LightningModule):
    def __init__(self, n_classes, n_layers, pretrained_model_name='camembert-base'):
        super(NestedNERModel, self).__init__()
        self.bert = CamembertModel.from_pretrained(pretrained_model_name)
        self.ner_layers = nn.ModuleList([nn.Linear(self.bert.config.hidden_size, n_classes) for _ in range(n_layers)])
        self.n_layers = n_layers

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs['last_hidden_state']
        all_logits = [ner_layer(sequence_output) for ner_layer in self.ner_layers]
        return all_logits
    
    def compute_loss(self, logits, labels):
        return sum(F.cross_entropy(logit.view(-1, logit.size(-1)), label.view(-1), ignore_index=-100) for logit, label in zip(logits, labels))

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, *labels = batch
        logits = self(input_ids, attention_mask)
        loss = self.compute_loss(logits, labels)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, *labels = batch
        logits = self(input_ids, attention_mask)
        val_loss = self.compute_loss(logits, labels)
        self.log('val_loss', val_loss, prog_bar=True, on_step=True, on_epoch=True)
        return val_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5)
        return optimizer