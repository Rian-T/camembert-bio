def tags_to_entities_with_offsets(tokens, tags, text):
    entities = []
    current_entity = None
    last_end_index = 0  # Add a variable to keep track of the last end index
    for i, (token, tag) in enumerate(zip(tokens, tags)):
        if tag.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {"type": tag[2:], "offsets": []}
            start_idx = text.index(token, last_end_index)  # Use last_end_index as the start position for searching
            end_idx = start_idx + len(token)
            current_entity["offsets"].append((start_idx, end_idx))
            last_end_index = end_idx  # Update last_end_index
        elif tag.startswith("I-") and current_entity:
            start_idx = text.index(token, last_end_index)  # Use last_end_index as the start position for searching
            end_idx = start_idx + len(token)
            current_entity["offsets"].append((start_idx, end_idx))
            last_end_index = end_idx  # Update last_end_index
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    if current_entity:
        entities.append(current_entity)
    
    return entities

