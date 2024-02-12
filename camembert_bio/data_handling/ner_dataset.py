
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import stanza

class NERDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
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

class NERDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data,
        val_data,
        test_data=None,
        tokenizer_name="camembert-base",
        batch_size=16,
        max_length=512,
        num_workers=0,
    ):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = NERDataset(
            self.train_data, self.tokenizer, self.max_length
        )
        self.val_dataset = NERDataset(
            self.val_data, self.tokenizer, self.max_length
        )
        if self.test_data is not None:
            self.test_dataset = NERDataset(
                self.test_data, self.tokenizer, self.max_length
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers) 


class NERPreprocessor:
    def __init__(self, data, lang="fr", granularity="smaller-preference"):
        self.data = data
        stanza.download(lang)
        self.nlp = stanza.Pipeline(lang=lang, processors="tokenize")
        self.label2id, self.id2label = self._label_mapping()
        self.granularity = granularity

    def _label_mapping(self):
        label_set = set()
        for split, items in self.data.items():
            for item in items:
                for ent in item["entities"]:
                    label_set.add(f"B-{ent['type']}")
                    label_set.add(f"I-{ent['type']}")

        label_list = sorted(
            list(label_set), key=lambda label: (label.split("-")[1], label.split("-")[0])
        )
        label_list.insert(0, "O")

        label2id = {label: i for i, label in enumerate(label_list)}
        id2label = {i: label for i, label in enumerate(label_list)}

        return label2id, id2label

    def process_data(self, split):
        processed_data = []
        for item in self.data[split]:
            text = item["passages"][0]["text"][0]

            # Tokenize the text with Stanza
            doc = self.nlp(text)

            for _, sentence in enumerate(doc.sentences):
                tokens = [token.text for token in sentence.tokens]
                offsets = [
                    (token.start_char, token.end_char) for token in sentence.tokens
                ]
                ner_tags = [self.label2id["O"]] * len(tokens)

                # Sort entities based on their lengths
                sorted_entities = sorted(
                    item["entities"],
                    key=lambda entity: entity["offsets"][0][1] - entity["offsets"][0][0],
                    reverse=False if self.granularity == "smaller-preference" else True,
                )

                for entity in sorted_entities:
                    start_char, end_char = entity["offsets"][0]
                    start_token, end_token = None, None

                    for i, (offset_start, offset_end) in enumerate(offsets):
                        if start_char in range(offset_start, offset_end + 1):
                            start_token = i
                        if end_char in range(offset_start, offset_end + 1):
                            end_token = i

                    if start_token is not None and end_token is not None:
                        ner_tags[start_token] = self.label2id[f"B-{entity['type']}"]
                        for i in range(start_token + 1, end_token + 1):
                            ner_tags[i] = self.label2id[f"I-{entity['type']}"]

                processed_data.append({"tokens": tokens, "ner_tags": ner_tags})

        return processed_data
