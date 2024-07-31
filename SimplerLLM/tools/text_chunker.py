from pydantic import BaseModel, Field
from typing import List
import re

import openai
import numpy as np

from SimplerLLM.language.llm import LLM as llm_genetation_instance
from SimplerLLM.language.embeddings import EmbeddingsLLM as llm_embeddings_instance

# from sklearn.metrics.pairwise import cosine_similarity


class ChunkInfo(BaseModel):
    text: str
    num_characters: int = Field(description="Number of characters in the chunk")
    num_words: int = Field(description="Number of words in the chunk")


class TextChunks(BaseModel):
    num_chunks: int = Field(description="Total number of chunks")
    chunk_list: List[ChunkInfo]


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

    return TextChunks(num_chunks=len(chunk_infos), chunk_list=chunk_infos)


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

    return TextChunks(num_chunks=len(chunk_infos), chunk_list=chunk_infos)


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

    return TextChunks(num_chunks=len(chunk_infos), chunk_list=chunk_infos)


def chunk_by_semantics(text: str,  llm_embeddings_instance: llm_embeddings_instance, threshold_percentage=90) -> TextChunks:

    # Split the input text into individual sentences.
    single_sentences_list = __split_sentences(text)

    # Combine adjacent sentences to form a context window around each sentence.
    combined_sentences = __combine_sentences(single_sentences_list)

    # Convert the combined sentences into vector representations using a neural network model.
    embeddings = __convert_to_vector(combined_sentences,llm_embeddings_instance=llm_embeddings_instance)

    # Calculate the cosine distances between consecutive combined sentence embeddings to measure similarity.
    distances = __calculate_cosine_similarities(embeddings)

    # Determine the threshold distance for identifying breakpoints based on the 80th percentile of all distances.
    breakpoint_percentile_threshold = threshold_percentage
    breakpoint_distance_threshold = np.percentile(
        distances, breakpoint_percentile_threshold
    )

    # Find all indices where the distance exceeds the calculated threshold, indicating a potential chunk breakpoint.
    indices_above_thresh = [
        i
        for i, distance in enumerate(distances)
        if distance > breakpoint_distance_threshold
    ]

    # Initialize the list of chunks and a variable to track the start of the next chunk.
    chunks = []
    start_index = 0

    # Loop through the identified breakpoints and create chunks accordingly.
    for index in indices_above_thresh:
        chunk = " ".join(single_sentences_list[start_index : index + 1])
        chunks.append(chunk)
        start_index = index + 1

    # If there are any sentences left after the last breakpoint, add them as the final chunk.
    if start_index < len(single_sentences_list):
        chunk = " ".join(single_sentences_list[start_index:])
        chunks.append(chunk)

    chunk_infos = [
        ChunkInfo(
            text=chunk.strip(),
            num_characters=len(chunk.strip()),
            num_words=len(chunk.strip().split()),
        )
        for chunk in chunks
        if chunk.strip()  # This condition filters out empty or whitespace-only paragraphs
    ]

    # Return the list of text chunks.
    return TextChunks(num_chunks=len(chunks), chunk_list=chunk_infos)




def __split_sentences(text):
    # Use regular expressions to split the text into sentences based on punctuation followed by whitespace.
    sentences = re.split(r"(?<=[.?!])\s+", text)
    return sentences




def __combine_sentences(sentences):
    # Create a buffer by combining each sentence with its previous and next sentence to provide a wider context.
    combined_sentences = []
    for i in range(len(sentences)):
        combined_sentence = sentences[i]
        if i > 0:
            combined_sentence = sentences[i - 1] + " " + combined_sentence
        if i < len(sentences) - 1:
            combined_sentence += " " + sentences[i + 1]
        combined_sentences.append(combined_sentence)
    return combined_sentences

def __convert_to_vector(combined_sentences_list, llm_embeddings_instance: llm_embeddings_instance):
    # Try to generate embeddings for a list of texts using a pre-trained model and handle any exceptions.
    try:
        response = llm_embeddings_instance.generate_embeddings(combined_sentences_list)
        #response = openai.embeddings.create(input=combined_sentences_list, model="text-embedding-3-small")
        embeddings = np.array([item.embedding for item in response])
        return embeddings
    except Exception as e:
        print("An error occurred:", e)
        return np.array([])  # Return an empty array in case of an error

def __calculate_cosine_similarities(embeddings):
    # Manually calculate the cosine similarities between consecutive embeddings.
    similarities = []
    for i in range(len(embeddings) - 1):
        vec1 = embeddings[i].flatten()
        vec2 = embeddings[i + 1].flatten()
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)

        if norm_vec1 == 0 or norm_vec2 == 0:
            # If either vector is zero, similarity is undefined (could also return 0)
            similarity = float("nan")
        else:
            similarity = dot_product / (norm_vec1 * norm_vec2)
        similarities.append(similarity)
    return similarities


