# Content Loading

Load text content from URLs, PDFs, Word documents, text files, and CSVs.

## Basic Usage

```python
from SimplerLLM.tools.generic_loader import load_content

# From a URL
doc = load_content("https://example.com/article")
print(doc.content)

# From a local file
doc = load_content("report.pdf")
print(doc.content)
```

## Supported Formats

| Format | Extensions | Parser |
|--------|-----------|--------|
| Web articles | URLs (`http://`, `https://`) | newspaper |
| PDF | `.pdf` | PyPDF2 |
| Word | `.docx` | python-docx |
| Text | `.txt` | Built-in |
| CSV (as text) | `.csv` | Built-in |
| Other | Any | Falls back to plain text |

> **Note:** For structured CSV data with row/column access, use `read_csv_file()` instead.

## Loading from URL

Extracts clean text from web articles and blog posts:

```python
from SimplerLLM.tools.generic_loader import load_content

doc = load_content("https://example.com/blog-post")

print(doc.title)            # Article title
print(doc.word_count)       # 1250
print(doc.content[:200])    # First 200 characters
```

## Loading from File

Works with PDF, Word, and text files:

```python
from SimplerLLM.tools.generic_loader import load_content

# PDF
doc = load_content("report.pdf")
print(f"{doc.word_count} words, {doc.file_size} bytes")

# Word document
doc = load_content("notes.docx")
print(doc.content)

# Text file
doc = load_content("data.txt")
print(doc.content)
```

## Loading CSV Files

For structured CSV access with row and column data:

```python
from SimplerLLM.tools.file_loader import read_csv_file

csv_doc = read_csv_file("data.csv")

print(csv_doc.row_count)        # 100
print(csv_doc.column_count)     # 5
print(csv_doc.content[0])       # First row (list of strings)
print(csv_doc.content[0][0])    # First cell
```

## Response Format

### TextDocument

Returned by `load_content()`:

```python
doc = load_content("report.pdf")

print(doc.content)           # The extracted text
print(doc.word_count)        # 1250
print(doc.character_count)   # 7500
print(doc.file_size)         # 45000 (bytes)
print(doc.title)             # Title (populated for URLs)
print(doc.url_or_path)       # Source path or URL
```

| Field | Type | Description |
|-------|------|-------------|
| `content` | `str` | The extracted text content |
| `word_count` | `int` | Number of words |
| `character_count` | `int` | Number of characters |
| `file_size` | `int` or `None` | File size in bytes |
| `title` | `str` or `None` | Document title (populated for URLs) |
| `url_or_path` | `str` or `None` | Source URL or file path |

### CSVDocument

Returned by `read_csv_file()`:

| Field | Type | Description |
|-------|------|-------------|
| `content` | `List[List[str]]` | Rows of CSV data |
| `row_count` | `int` | Number of rows |
| `column_count` | `int` | Number of columns |
| `total_fields` | `int` | Total number of fields across all rows |
| `file_size` | `int` or `None` | File size in bytes |
| `url_or_path` | `str` or `None` | File path |
