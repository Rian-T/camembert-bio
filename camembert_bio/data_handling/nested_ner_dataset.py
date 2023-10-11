import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pytorch_lightning as pl
from torch.utils.data import DataLoader


class NestedNERDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["tokens"]
        labels_layers = [item[key] for key in item.keys() if "ner_tags" in key]

        encoding = self.tokenizer(
            text,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        labels_tensors = []
        for labels in labels_layers:
            adjusted_labels = []
            for word_idx in encoding.word_ids(batch_index=0):
                if word_idx is None:
                    adjusted_labels.append(-100)
                else:
                    adjusted_labels.append(labels[word_idx])
            labels_tensors.append(torch.tensor(adjusted_labels[: self.max_length]))

        return (input_ids, attention_mask, *labels_tensors)


class NestedNERDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data,
        val_data,
        tokenizer_name="camembert-base",
        batch_size=16,
        max_length=512,
    ):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.batch_size = batch_size
        self.max_length = max_length

    def setup(self, stage=None):
        self.train_dataset = NestedNERDataset(
            self.train_data, self.tokenizer, self.max_length
        )
        self.val_dataset = NestedNERDataset(
            self.val_data, self.tokenizer, self.max_length
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

class NestedNERPreprocessor:
    def __init__(self, data, tokenizer_name):
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.label2id, self.id2label = self._label_mapping()
        self.n_layers = len(self.label2id)

    def _label_mapping(self):
        label_set = set()
        label_count = {}
        for split, items in self.data.items():
            for item in items:
                for ent in item["entities"]:
                    label_set.add(ent["type"])
                    label_count[ent["type"]] = label_count.get(ent["type"], 0) + 1
        label_set = sorted(label_set, key=lambda label: label_count[label], reverse=True)
        label2id = {label: i for i, label in enumerate(label_set)}
        id2label = {i: label for i, label in enumerate(label_set)}
        return label2id, id2label

    def _determine_layer(self, entity):
        return self.label2id[entity['type']]

    def process_data(self, split):
        processed_data = []
        for item in self.data[split]:
            text = item['passages'][0]['text'][0]
            inputs_with_offsets = self.tokenizer(text, return_offsets_mapping=True)
            tokens = inputs_with_offsets.tokens()[1:]
            offsets = inputs_with_offsets["offset_mapping"][1:]
            aligned_labels = [[0] * len(tokens) for _ in range(self.n_layers)]
            for entity in item["entities"]:
                start_char, end_char = entity['offsets'][0]
                start_token, end_token = None, None
                for i, (offset_start, offset_end) in enumerate(offsets):
                    if start_char == 0:
                        start_token = 0
                    elif start_char >= offset_start and start_char < offset_end:
                        start_token = i
                    if end_char > offset_start and end_char <= offset_end:
                        end_token = i
                if start_token is None or end_token is None:
                    print(f"Could not map entity {entity['text']} at position {start_char}-{end_char} to tokens")
                    continue
                layer_idx = self._determine_layer(entity)
                if layer_idx >= self.n_layers:
                    continue
                aligned_labels[layer_idx][start_token] = 1
                if start_token != end_token:
                    aligned_labels[layer_idx][start_token+1:end_token+1] = [2] * (end_token - start_token)
            ner_tags_dict = {f'ner_tags_layer{i+1}': aligned_labels[i] for i in range(self.n_layers)}
            processed_data.append({
                'tokens': tokens,
                **ner_tags_dict
            })
        return processed_data