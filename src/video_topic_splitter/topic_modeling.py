"""Topic modeling and text analysis functionality."""

import os
import json
import spacy
import progressbar
from gensim import corpora
from gensim.models import LdaMulticore
from gensim.models.phrases import Phrases, Phraser
import os
from openai import OpenAI

from .constants import CHECKPOINTS
from .project import save_checkpoint

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

import os

def preprocess_text(text, custom_stopwords=None, stopwords_file_path="/tmp/custom_stop_words.txt"):
    """Preprocess text for topic modeling."""
    print("Preprocessing text...")
    doc = nlp(text)
    
    if stopwords_file_path:
        try:
            with open(stopwords_file_path, 'r') as f:
                file_stopwords = [line.strip() for line in f]
                if custom_stopwords:
                    custom_stopwords.extend(file_stopwords)
                else:
                    custom_stopwords = file_stopwords
        except FileNotFoundError:
            print(f"Stopwords file not found at: {stopwords_file_path}")
        except Exception as e:
            print(f"Error reading stopwords file: {e}")

    sentences = []
    for sent in doc.sents:
        sentence = []
        for token in sent:
            if token.ent_type_:
                sentence.append(get_compound_subject(token))
            elif "subj" in token.dep_:
                if token.dep_ in ["nsubj", "nsubjpass", "csubj"]:
                    sentence.append(get_compound_subject(token))
            elif not token.is_stop and not token.is_punct and token.is_alpha and (custom_stopwords is None or token.lemma_.lower() not in custom_stopwords):
                sentence.append(token.lemma_.lower())
        sentences.append(sentence)

    cleaned_sentences = [list(s) for s in set(tuple(sent) for sent in sentences) if s]

    # Create bigram and trigram models
    bigram_model = Phrases(cleaned_sentences, min_count=1, threshold=1)
    bigram_phraser = Phraser(bigram_model)
    trigram_model = Phrases(bigram_phraser[cleaned_sentences], min_count=1, threshold=1)
    trigram_phraser = Phraser(trigram_model)

    # Apply the models
    sentences_with_bigrams = [bigram_phraser[sent] for sent in cleaned_sentences]
    sentences_with_trigrams = [trigram_phraser[sent] for sent in sentences_with_bigrams]

    print(f"Text preprocessing complete. Extracted {len(sentences_with_trigrams)} unique sentences with bigrams and trigrams.")
    return sentences_with_trigrams

def get_compound_subject(token):
    """Extract compound subjects from a token."""
    subject = [token.text]
    for left_token in token.lefts:
        if left_token.dep_ == "compound":
            subject.insert(0, left_token.text)
    for right_token in token.rights:
        if right_token.dep_ == "compound":
            subject.append(right_token.text)
    return " ".join(subject)

def perform_topic_modeling(subjects, num_topics=5):
    """Perform LDA topic modeling on preprocessed subjects."""
    print(f"Performing topic modeling with {num_topics} topics...")
    dictionary = corpora.Dictionary(subjects)
    corpus = [dictionary.doc2bow(subject) for subject in subjects]
    lda_model = LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=100,
        chunksize=100,
        passes=10,
        per_word_topics=True,
    )
    print("Topic modeling complete.")
    return lda_model, corpus, dictionary

def perform_topic_modeling_openrouter(text, num_topics=5):
    """Perform topic modeling using OpenRouter API and microsoft/phi-4 model."""
    """Perform topic modeling using OpenRouter API and microsoft/phi-4 model with openai library."""
    print("Performing topic modeling using OpenRouter API with openai library...")
    api_key = os.getenv("OPENROUTER_API_KEY")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://video-topic-splitter.example",  # Replace with your site URL if needed
                "X-Title": "Video Topic Splitter",  # Replace with your site title if needed
            },
            model="microsoft/phi-4",
            messages=[
                {
                    "role": "user",
                    "content": f"Identify the main topics in the following text and provide keywords for each topic:\\n\\n{text}"
                }
            ]
        )
        topics_data = completion.choices[0].message.content
        print("OpenRouter API response received.")
        return topics_data  # Return raw response content for now - needs parsing

    except Exception as e:
        print(f"Error during OpenRouter API request: {e}")
        return None  # Or raise exception

def identify_segments(transcript, lda_model, dictionary, num_topics):
    """Identify video segments based on topic analysis."""
    print("Identifying segments based on topics...")
    segments = []
    current_segment = {"start": 0, "end": 0, "content": "", "topic": None}
    
    preprocessed_sentences = preprocess_text(" ".join([sentence["content"] for sentence in transcript]))
    
    sentence_index = 0
    for sentence in progressbar.progressbar(transcript):
        if not preprocessed_sentences:
            continue
        
        if sentence_index >= len(preprocessed_sentences):
            sentence_index = len(preprocessed_sentences) - 1
        
        bow = dictionary.doc2bow(preprocessed_sentences[sentence_index])
        topic_dist = lda_model.get_document_topics(bow)
        dominant_topic = max(topic_dist, key=lambda x: x[1])[0] if topic_dist else None

        if dominant_topic != current_segment["topic"]:
            if current_segment["content"]:
                current_segment["end"] = sentence["start"]
                segments.append(current_segment)
            current_segment = {
                "start": sentence["start"],
                "end": sentence["end"],
                "content": sentence["content"],
                "topic": dominant_topic,
            }
        else:
            current_segment["end"] = sentence["end"]
            current_segment["content"] += " " + sentence["content"]
        
        sentence_index += 1

    if current_segment["content"]:
        segments.append(current_segment)

    print(f"Identified {len(segments)} segments.")
    return segments

def generate_metadata(segments, lda_model):
    """Generate metadata for identified segments."""
    print("Generating metadata for segments...")
    metadata = []
    for i, segment in enumerate(progressbar.progressbar(segments)):
        segment_metadata = {
            "segment_id": i + 1,
            "start_time": segment["start"],
            "end_time": segment["end"],
            "duration": segment["end"] - segment["start"],
            "dominant_topic": segment["topic"],
            "top_keywords": [
                word for word, _ in lda_model.show_topic(segment["topic"], topn=5)
            ],
            "transcript": segment["content"],
        }
        metadata.append(segment_metadata)
    print("Metadata generation complete.")
    return metadata

def process_transcript(transcript, project_path, num_topics=5):
    """Process transcript for topic modeling and segmentation."""
    full_text = " ".join([sentence["content"] for sentence in transcript])
    preprocessed_subjects = preprocess_text(full_text)
    
    # LDA Topic Modeling (Old implementation - Commented out)
    # lda_model, corpus, dictionary = perform_topic_modeling(preprocessed_subjects, num_topics)    
    # save_checkpoint(project_path, CHECKPOINTS['TOPIC_MODELING_COMPLETE'], {
    #     'lda_model': lda_model,
    #     'corpus': corpus,
    #     'dictionary': dictionary
    # })
    # segments = identify_segments(transcript, lda_model, dictionary, num_topics)
    # metadata = generate_metadata(segments, lda_model)

    # OpenRouter Topic Modeling (New implementation)
    openrouter_topics = perform_topic_modeling_openrouter(full_text, num_topics)
    # Placeholder for segment identification and metadata generation using OpenRouter topics
    segments = []  # Placeholder
    metadata = []  # Placeholder


    results = {
        "topics": [
            {
                "topic_id": topic_id,
                "words": ["word1", "word2", "word3"]  # Placeholder - needs to be updated based on OpenRouter response
            }
            for topic_id in range(num_topics)
        ],
        "segments": metadata,
    }

    results_path = os.path.join(project_path, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")

    return results
