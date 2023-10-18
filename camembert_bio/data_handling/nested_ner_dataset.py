import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import stanza


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
        num_workers=0,
    ):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = NestedNERDataset(
            self.train_data, self.tokenizer, self.max_length
        )
        self.val_dataset = NestedNERDataset(
            self.val_data, self.tokenizer, self.max_length
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class NestedPerClassNERPreprocessor:
    def __init__(self, data, lang):
        self.data = data
        stanza.download(lang)
        self.nlp = stanza.Pipeline(lang=lang, processors="tokenize")
        self.label2id, self.id2label = self._label_mapping()
        self.n_layers = len(self.label2id)

    def _label_mapping(self):
        label_set = set()
        for split, items in self.data.items():
            for item in items:
                for ent in item["entities"]:
                    label_set.add(ent["type"])
        label_list = list(label_set)
        label_list.sort()
        label2id = {label: i for i, label in enumerate(label_list)}
        id2label = {i: label for i, label in enumerate(label_list)}
        return label2id, id2label

    def _determine_layer(self, entity):
        return self.label2id[entity["type"]]

    def process_data(self, split):
        processed_data = []
        for item in self.data[split]:
            text = item["passages"][0]["text"][0]

            # Tokenize the text with Stanza
            doc = self.nlp(text)
            sentences = [sent.text for sent in doc.sentences]
            sentence_offsets = [text.index(sent) for sent in sentences]

            for idx, sentence in enumerate(doc.sentences):
                tokens = [token.text for token in sentence.tokens]
                offsets = [
                    (token.start_char, token.end_char) for token in sentence.tokens
                ]

                sentence_start_offset = sentence_offsets[idx]
                sentence_end_offset = sentence_start_offset + len(sentence.text)

                # Recalculate the entities' offsets for the current sentence
                current_entities = []
                for e in item["entities"]:
                    entity_text = e["text"][0].strip(" ,.")
                    entity_start = text.find(
                        entity_text, sentence_start_offset, sentence_end_offset
                    )
                    if entity_start != -1:  # if entity is found within the sentence
                        entity_end = entity_start + len(entity_text)
                        if (
                            entity_start >= sentence_start_offset
                            and entity_end <= sentence_end_offset
                        ):
                            print(
                                f"found entity {entity_text} at position {entity_start}-{entity_end}"
                            )
                            current_entities.append(
                                {
                                    "text": entity_text,
                                    "offsets": [(entity_start, entity_end)],
                                    "type": e["type"],
                                }
                            )

                aligned_labels = [[0] * len(tokens) for _ in range(self.n_layers)]
                for entity in current_entities:
                    start_char, end_char = entity["offsets"][0]
                    start_token, end_token = None, None
                    for i, (offset_start, offset_end) in enumerate(offsets):
                        if start_char in range(
                            offset_start - 1, offset_end + 1
                        ):  # tolerance window
                            start_token = i
                        if end_char in range(
                            offset_start - 1, offset_end + 1
                        ):  # tolerance window
                            end_token = i
                    if start_token is None or end_token is None:
                        print(f"Entity: {entity['text']} ({start_char}-{end_char})")
                        print("Tokens and Offsets:", list(zip(tokens, offsets)))
                        print(
                            f"Could not map entity {entity['text']} at position {start_char}-{end_char} to tokens"
                        )
                        continue
                    layer_idx = self._determine_layer(entity)
                    if layer_idx >= self.n_layers:
                        continue
                    aligned_labels[layer_idx][start_token] = 1
                    if start_token != end_token:
                        aligned_labels[layer_idx][start_token + 1 : end_token + 1] = [
                            2
                        ] * (end_token - start_token)
                ner_tags_dict = {
                    f"ner_tags_layer{i+1}": aligned_labels[i]
                    for i in range(self.n_layers)
                }
                processed_data.append({"tokens": tokens, **ner_tags_dict})

        return processed_data


class NestedPerDepthNERPreprocessor:
    def __init__(self, data, lang="fr"):
        self.data = data
        stanza.download(lang)
        self.nlp = stanza.Pipeline(lang=lang, processors="tokenize")
        self.label2id, self.id2label, self.n_layers = self._label_mapping()

    def _label_mapping(self):
        label_set = set()

        for split, items in self.data.items():
            for item in items:
                for ent in item["entities"]:
                    label_set.add(f"B-{ent['type']}")
                    label_set.add(f"I-{ent['type']}")

        label_list = list(label_set)

        # sort the list without considering the tag, and then prepend 'O'
        label_list = sorted(
            label_list, key=lambda label: (label.split("-")[1], label.split("-")[0])
        )
        label_list.insert(0, "O")

        label2id = {label: i for i, label in enumerate(label_list)}
        id2label = {i: label for i, label in enumerate(label_list)}

        # Compute the maximum depth in the entire dataset
        max_depth = 0
        for split in self.data:
            for item in self.data[split]:
                depths = self._compute_depth(item["entities"])
                if len(depths) == 0:
                    continue
                max_depth = max(max_depth, max(depths.values()))

        return label2id, id2label, max_depth

    def _compute_depth(self, entities):
        """
        Compute the depth of each entity based on nesting.
        Returns a dictionary with entity offsets as keys and depth as values.
        """
        depths = {}
        for entity in entities:
            start, end = entity["offsets"][0]
            if (start, end) not in depths:
                depths[(start, end)] = 1  # or any default value
            # Exclude discontinuous entities
            if len(entity["offsets"]) > 1:
                continue
            start, end = entity["offsets"][0]
            depth = 1
            for other_entity in entities:
                if entity == other_entity:  # Avoid comparing the entity with itself
                    continue
                other_start, other_end = other_entity["offsets"][0]
                if other_start <= start and other_end >= end:
                    depth += 1
            depths[(start, end)] = depth
        return depths

    def _determine_layer(self, entity):
        """
        The layer is simply the depth of the entity.
        """
        return entity["depth"] - 1

    def process_data(self, split):
        processed_data = []
        for item in self.data[split]:
            text = item["passages"][0]["text"][0]

            # Tokenize the text with Stanza
            doc = self.nlp(text)
            sentences = [sent.text for sent in doc.sentences]
            sentence_offsets = [text.index(sent) for sent in sentences]

            remaining_entities = item["entities"].copy()

            for idx, sentence in enumerate(doc.sentences):
                tokens = [token.text for token in sentence.tokens]
                offsets = [
                    (token.start_char, token.end_char) for token in sentence.tokens
                ]
                sentence_start_offset = sentence_offsets[idx]
                sentence_end_offset = sentence_start_offset + len(sentence.text)
                # Filter entities that belong to the current sentence
                current_entities = [
                    e
                    for e in remaining_entities
                    if e["offsets"][0][0] >= sentence_start_offset
                    and e["offsets"][0][1] <= sentence_end_offset
                ]

                # Remove these entities from the remaining_entities list
                remaining_entities = [
                    e for e in remaining_entities if e not in current_entities
                ]

                # Calculate depth for each entity
                depths = self._compute_depth(current_entities)
                for entity in current_entities:
                    if tuple(entity["offsets"][0]) in depths:
                        entity["depth"] = depths[tuple(entity["offsets"][0])]
                    else:
                        entity["depth"] = 1  # or any default value

                aligned_labels = [
                    [self.label2id["O"]] * len(tokens) for _ in range(self.n_layers)
                ]
                for entity in current_entities:
                    start_char, end_char = entity["offsets"][0]
                    start_token, end_token = None, None
                    for i, (offset_start, offset_end) in enumerate(offsets):
                        if start_char in range(
                            offset_start - 1, offset_end + 1
                        ):  # tolerance window
                            start_token = i
                        if end_char in range(
                            offset_start - 1, offset_end + 1
                        ):  # tolerance window
                            end_token = (
                                i - 1
                            )  # -1 because the end token is the token before the end offset

                    if start_token is None or end_token is None:
                        continue
                    layer_idx = self._determine_layer(entity)
                    if layer_idx >= self.n_layers:
                        continue
                    aligned_labels[layer_idx][start_token] = self.label2id[
                        f"B-{entity['type']}"
                    ]
                    if start_token != end_token:
                        aligned_labels[layer_idx][start_token + 1 : end_token + 1] = [
                            self.label2id[f"I-{entity['type']}"]
                        ] * (end_token - start_token)
                ner_tags_dict = {
                    f"ner_tags_layer{i+1}": aligned_labels[i]
                    for i in range(self.n_layers)
                }
                processed_data.append({"tokens": tokens, **ner_tags_dict})

        return processed_data
