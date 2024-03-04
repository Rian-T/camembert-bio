import re
from datasets import load_dataset
from camembert_bio.data_handling.ner_dataset import NERPreprocessor, NERDataModule
import copy
from camembert_bio.data_handling.nested_ner_dataset import (
    NestedPerDepthNERPreprocessor,
    NestedNERDataModule,
)

data = load_dataset("bigbio/bc5cdr", "bc5cdr_bigbio_kb")


for i in range(100):

    text = ""
    for p in data["train"][i]["passages"]:
        text += " ".join(p["text"])
    
    entities = []
    for entity in data["train"][i]["entities"]:
        entity_text = entity["text"]
        entity_offset = entity["offsets"][0]
        entities.append((entity_text, entity_offset))
    predicted_text = "@@Stroke## associated with @@cocaine## use. We describe eight patients in whom @@ cocaine         ## use was related to @@ stroke ##and review 39 cases from the literature. Among these 47 patients the mean (+/- SD) age was 32.5 +/- 12.1 years; 76% (34/45) were men. Stroke followed cocaine use by inhalation, intranasal, intravenous, and intramuscular routes. Intracranial aneurysms or arteriovenous malformations were present in 17 of 32 patients studied angiographically or at autopsy; cerebral vasculitis was present in two patients. Cerebral infarction occurred in 10 patients (22%), intracerebral hemorrhage in 22 (49%), and subarachnoid hemorrhage in 13 (29%). These data indicate that (1) the apparent incidence of stroke related to cocaine use is increasing; (2) cocaine-associated stroke occurs primarily in young adults; (3) stroke may follow any route of cocaine administration; (4) stroke after cocaine use is frequently associated with intracranial aneurysms and arteriovenous malformations; and (5) in cocaine-associated stroke, the frequency of intracranial hemorrhage exceeds that of cerebral infarction."
    

    def parse_entities_offsets_from_tags(text, start_tag="@@", end_tag="##"):
        """
        This function finds the offsets of entities in a given text. 
        The entities are marked with a start tag at the start and an end tag at the end.

        Args:
            text (str): The input text with marked entities.
            start_tag (str, optional): The tag that marks the start of an entity. Defaults to "@@".
            end_tag (str, optional): The tag that marks the end of an entity. Defaults to "##".

        Returns:
            list: A list of tuples where each tuple contains an entity and its start and end offsets.
        """
        # Remove spaces around entities
        text = re.sub(rf"{start_tag}\s*(.*?){end_tag}", lambda match: start_tag + match.group(1).strip() + end_tag, text)

        entities = []
        total_entities = 0
        start = 0

        while True:
            start = text.find(start_tag, start)
            if start == -1:  # No more entities
                break

            end = text.find(end_tag, start)
            entity = text[start+len(start_tag):end]  # Exclude the start and end tags from the entity

            # Adjust the offsets to ignore the start and end tags
            start_adjusted = start - 2*total_entities*len(start_tag)
            end_adjusted = end - 2*total_entities*len(end_tag)

            entities.append((entity, start_adjusted, end_adjusted))

            start = end + len(end_tag)  # Move past this entity
            total_entities += 1

        return entities

    # print(find_entity_offsets(predicted_text) == [(entity_text, entity_offset[0], entity_offset[1])])
    # print(all([predicted_text[start:end] == entity for entity, start, end in find_entity_offsets(predicted_text)]))    
    # print("text:", text)
    # print("predicted_text:", predicted_text)
    # for entity_text, entity_offset in entities:
    #     print("entity_text:", entity_text)
    #     print("entity_offset:", entity_offset)
    # print("predicted_entity_offsets:", find_entity_offsets(predicted_text))
    # print(i)
    # print("="*80)

# pretty print everything
print("text:", text)
print("entity_text:", entity_text)
print("entity_offset:", entity_offset)
print("predicted_text:", predicted_text)