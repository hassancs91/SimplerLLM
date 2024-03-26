from pydantic import BaseModel, Field
from typing import List
import re


class ChunkInfo(BaseModel):
    text: str
    num_characters: int = Field(description="Number of characters in the chunk")
    num_words: int = Field(description="Number of words in the chunk")


class TextChunks(BaseModel):
    num_chunks: int = Field(description="Total number of chunks")
    chunks: List[ChunkInfo]


def chunk_by_max_chunk_size(
    text: str, max_chunk_size: int, preserve_sentence_structure: bool = False
) -> TextChunks:
    """
    Split the given text into chunks based on the maximum chunk size trying to reserve sentence endings if preserve_sentence_structure is enabled

    Parameters:
    - text (str): The input text to be split into chunks.
    - max_chunk_size (int): The maximum size of each chunk.
    - preserve_sentence_structure: Whether to consider preserving the sentence structure when splitting the text.

    Returns:
    - TextChunks: An object containing the total number of chunks and a list of ChunkInfo objects.
    - num_chunks (int): The total number of chunks.
    - chunks (List[ChunkInfo]): A list of ChunkInfo objects, each representing a chunk of the text.
        - chunk (str): The chunk of text.
        - num_characters (int): The number of characters in the chunk.
        - num_words (int): The number of words in the chunk.
    """

    if preserve_sentence_structure:
        sentences = re.split(r"(?<=[.!?]) +", text)
    else:
        sentences = [
            text[i : i + max_chunk_size] for i in range(0, len(text), max_chunk_size)
        ]

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if preserve_sentence_structure:
            if len(current_chunk) + len(sentence) <= max_chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = sentence
                else:
                    # If the current chunk is empty and the sentence is longer than the max size,
                    # accept this sentence as a single chunk even if it exceeds the max size.
                    chunks.append(sentence)
                    sentence = ""
        else:
            chunks.append(sentence)

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    chunk_infos = [
        ChunkInfo(text=chunk, num_characters=len(chunk), num_words=len(chunk.split()))
        for chunk in chunks
    ]

    return TextChunks(num_chunks=len(chunk_infos), chunks=chunk_infos)


def chunk_by_sentences(text: str) -> TextChunks:
    # Regular expression for splitting by sentence
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", text)

    chunk_infos = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:  # This condition filters out empty or whitespace-only sentences
            chunk_info = ChunkInfo(
                text=sentence,
                num_characters=len(sentence),
                num_words=len(sentence.split()),
            )
            chunk_infos.append(chunk_info)

    return TextChunks(num_chunks=len(chunk_infos), chunks=chunk_infos)


def chunk_by_paragraphs(text: str) -> TextChunks:
    # Splitting the text into paragraphs
    paragraphs = re.split(r"(?<=\n)(?=[A-Z0-9])", text)

    chunk_infos = [
        ChunkInfo(
            text=paragraph.strip(),
            num_characters=len(paragraph.strip()),
            num_words=len(paragraph.strip().split()),
        )
        for paragraph in paragraphs
        if paragraph.strip()  # This condition filters out empty or whitespace-only paragraphs
    ]

    return TextChunks(num_chunks=len(chunk_infos), chunks=chunk_infos)
