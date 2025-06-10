
from pymongo import MongoClient
from collections import defaultdict
import sys

# MongoDB connection
connection_string = "mongodb://root:SMgM242EzXWK8RV5eqjWGl2Yg61NMHoQkaVPIIE2zyXjhWYDVwbv0OjBWxe67n0k@62.171.145.173:5432/?directConnection=true"
client = MongoClient(connection_string)
db = client["test"]
collection = db["youtube_title_framworks"]

# Fetch all documents
documents = list(collection.find())
print(f"Total documents: {len(documents)}")

# Function to remove duplicates
def remove_duplicates(duplicate_groups, field_name):
    if not duplicate_groups:
        print(f"No {field_name} duplicates to remove.")
        return 0
    
    removed_count = 0
    for field_value, docs in duplicate_groups.items():
        if len(docs) <= 1:
            continue
            
        # Sort by Hook Score in descending order
        sorted_docs = sorted(docs, key=lambda x: x.get('Hook Score', 0), reverse=True)
        
        # Keep the one with highest Hook Score
        keep_doc = sorted_docs[0]
        docs_to_remove = sorted_docs[1:]
        
        print(f"\nKeeping {field_name}: '{field_value}' with Hook Score: {keep_doc.get('Hook Score', 0)}")
        
        # Remove the duplicates
        for doc in docs_to_remove:
            collection.delete_one({"_id": doc["_id"]})
            print(f"  - Removed duplicate with Hook Score: {doc.get('Hook Score', 0)}")
            removed_count += 1
    
    return removed_count

# Check for Framework duplicates
framework_groups = defaultdict(list)
for doc in documents:
    framework = doc.get("Framework", "")
    framework_groups[framework].append(doc)

# Display Framework duplicates
framework_duplicates = {k: v for k, v in framework_groups.items() if len(v) > 1}
print(f"\nFound {len(framework_duplicates)} duplicate Framework groups:")
for framework, docs in framework_duplicates.items():
    print(f"\nFramework: '{framework}'")
    for doc in docs:
        print(f"  - Title: '{doc.get('Title', '')}', Hook Score: {doc.get('Hook Score', 0)}")

# Check for Title duplicates
title_groups = defaultdict(list)
for doc in documents:
    title = doc.get("Title", "")
    title_groups[title].append(doc)

# Display Title duplicates
title_duplicates = {k: v for k, v in title_groups.items() if len(v) > 1}
print(f"\nFound {len(title_duplicates)} duplicate Title groups:")
for title, docs in title_duplicates.items():
    print(f"\nTitle: '{title}'")
    for doc in docs:
        print(f"  - Framework: '{doc.get('Framework', '')}', Hook Score: {doc.get('Hook Score', 0)}")

# Ask for confirmation to remove duplicates
print("\n" + "="*50)
print("Do you want to remove duplicates? This will keep the version with the highest Hook Score for each duplicate group.")
print("Type 'yes-framework' to remove Framework duplicates")
print("Type 'yes-title' to remove Title duplicates")
print("Type 'yes-both' to remove both types of duplicates")
print("Type anything else to exit without removing duplicates")
print("="*50)

user_input = input("> ").strip().lower()

removed_count = 0
if user_input == "yes-framework":
    removed_count = remove_duplicates(framework_duplicates, "Framework")
    print(f"\nRemoved {removed_count} Framework duplicates.")
elif user_input == "yes-title":
    removed_count = remove_duplicates(title_duplicates, "Title")
    print(f"\nRemoved {removed_count} Title duplicates.")
elif user_input == "yes-both":
    framework_removed = remove_duplicates(framework_duplicates, "Framework")
    # Refresh the documents list and title duplicates after removing framework duplicates
    documents = list(collection.find())
    title_groups = defaultdict(list)
    for doc in documents:
        title = doc.get("Title", "")
        title_groups[title].append(doc)
    title_duplicates = {k: v for k, v in title_groups.items() if len(v) > 1}
    title_removed = remove_duplicates(title_duplicates, "Title")
    removed_count = framework_removed + title_removed
    print(f"\nRemoved {framework_removed} Framework duplicates and {title_removed} Title duplicates.")
else:
    print("\nExiting without removing any duplicates.")

# Close MongoDB connection
client.close()

if removed_count > 0:
    print(f"\nSuccessfully removed {removed_count} duplicate(s).")
    print("Run the script again to verify that duplicates have been removed.")
