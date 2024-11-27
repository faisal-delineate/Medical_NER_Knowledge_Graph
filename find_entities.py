import re

all_relationship_data='outputs/paper_16_entity_relationships.txt'
# Given text
with open(all_relationship_data, 'r', encoding='utf-8') as f:
    text = f.read()


# Regex pattern to extract entities
entity_type_pattern = r"'[^']+', '([^']+)'\)"
entity_types = re.findall(entity_type_pattern, text)

# Remove duplicates by converting to a set, then back to a list
unique_entity_types = list(set(entity_types))

# Display the result
print(unique_entity_types)
