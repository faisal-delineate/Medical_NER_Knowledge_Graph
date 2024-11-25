import re
all_relationship_data='all_relationship_data.txt'
# Given text
with open(all_relationship_data, 'r', encoding='utf-8') as f:
    text = f.read()

# Regex to find all relationships
regex = r'"relationship":\s*"(.*?)"'

# Find all matches
relationships = re.findall(regex, text)

# Print results
print("Extracted Relationships:")
print(list(set(relationships)))
