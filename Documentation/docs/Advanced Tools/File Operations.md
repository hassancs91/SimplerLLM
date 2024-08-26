---
sidebar_position: 1
--- 

# File Operations

SimplerLLM supports creating and loading the content of various file types. This makes it easy to load the content of any file or even content from the internet using a generic function.

The file operations available are categorized into three primary areas:
- **Saving Text to Files**: This functionality allows for the writing of text data to files ensuring errors are handled correctly.
- **Loading CSV Files**: This functionality allows easy reading of any CSV file document, where it returns specific strucutred data.
- **Generic File Loading**: This includes loading the details of various types of files, such as plain text, PDFs, DOCX, web pages, and even youtube video data. 

Here's how each of them works:

## Saving Text to File

This operation contains a single function `save_text_to_file` which takes as input the text you want to save and the name of the file you want to save in.

If the file is already present in your directory it just rewrites its content, however if it's not present it creates the file and adds the the input text to it. Here's an example:

```python
from SimplerLLM.tools.file_functions import save_text_to_file

input_text = save_text_to_file("This is the text saved in the file", "file.txt")

print(input_text)
```

As you can see it takes 2 paramters:
- `text (str)`: The text content to save.
- `filename (str)` (Optional): The destination filename. Defaults to "output.txt".

Then, it returns a bool (True/False), representing if the file was created successfully or not.

## Loading CSV Files

This operation also contains a single function `load_csv_file` which takes as input the path to the CSV, and returns a `CSVDocument` object which provides a structured way to access the CSV data, including the following attributes that you can access independently:
- `file_size`: The size of the CSV file in bytes.
- `row_count`: Number of rows in the CSV.
- `column_count`: Number of columns in the CSV.
- `total_fields`: Total number of data fields.
- `content`: Nested list representing rows and columns.
- `title`: Title of the document (Will be set to None in this function)
- `url_or_path`: CSV file name.

Here's an example of the function in action:

```python
from SimplerLLM.tools.file_loader import read_csv_file

csv_data = read_csv_file("text.csv")

print(csv_data)
```

When you print the csv_data as is it will return the whole `CSVDocument` object with all its attributes. However, if you want access for example only the content of the file, here's how you do it:

```python
from SimplerLLM.tools.file_loader import read_csv_file

csv_data = read_csv_file("text.csv")

print(csv_data.content)
```

Use the same method for accessing the other attributes. 
Here's another example on how to access the column count:

```python
from SimplerLLM.tools.file_loader import read_csv_file

csv_data = read_csv_file("text.csv")

print(csv_data.column_count)
```

## Generic Loading Of Other File Types

This generic loader supports a ton of file types which are:
- Web Articles
- YouTube video transcripts
- Traditional formats like TXT, PDF, CSV, and DOCX.

The `load_content` function takes the file name as input, and returns a `Text Document` object that has the following attributes:
- `file_size`: The size of the file in bytes.
- `word_count`: The number of words in the file.
- `character_count`: The number of characters in the file.
- `content`: String representing the contents of the file.
- `title`: Title of the document (if it has one)
- `url_or_path`: file name.

Here's an example of the function in action:

```python
from SimplerLLM.tools.generic_loader import load_content

file_data = load_content("file_name.csv")

print(file_data)
```

When you print the file_data as is it will return the whole `Text Document` object with all its attributes. However, if you want access for example only the content of the file, here's how you do it:

```python
from SimplerLLM.tools.generic_loader import load_content

file_data = load_content("file_name.csv")

print(file_data.content)
```

Use the same method for accessing the other attributes. 
Here's another example on how to access the word count:

```python
from SimplerLLM.tools.generic_loader import load_content

file_data = load_content("file_name.csv")

print(file_data.word_count)
```

That's how you can benefit from SimplerLLM to make interaction with files Simpler!