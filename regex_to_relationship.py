import re
all_relationship_data='outputs/paper_16_entity_relationships.txt'
# Given text
with open(all_relationship_data, 'r', encoding='utf-8') as f:
    text = f.read()

relationship_pattern = r"'relationship':\s*'([^']+)'"  # Matches the relationship values
relationships = re.findall(relationship_pattern, text)

# Remove duplicates by converting to a set, then back to a list
unique_relationships = list(set(relationships))

# Display the result
print(unique_relationships)
