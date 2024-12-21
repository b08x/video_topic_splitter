"""Topic modeling and text analysis functionality."""

import os
import json
import spacy
import progressbar
from gensim import corpora
from gensim.models import LdaMulticore

from .constants import CHECKPOINTS
from .project import save_checkpoint

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    """Preprocess text for topic modeling."""
    print("Preprocessing text...")
    doc = nlp(text)
    subjects = []

    for sent in doc.sents:
        for token in sent:
            if "subj" in token.dep_:
                if token.dep_ in ["nsubj", "nsubjpass", "csubj"]:
                    subject = get_compound_subject(token)
                    subjects.append(subject)

    cleaned_subjects = [
        [
            token.lemma_.lower()
            for token in nlp(subject)
            if not token.is_stop and not token.is_punct and token.is_alpha
        ]
        for subject in subjects
    ]

    cleaned_subjects = [
        list(s) for s in set(tuple(sub) for sub in cleaned_subjects) if s
    ]

    print(f"Text preprocessing complete. Extracted {len(cleaned_subjects)} unique subjects.")
    return cleaned_subjects

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

def identify_segments(transcript, lda_model, dictionary, num_topics):
    """Identify video segments based on topic analysis."""
    print("Identifying segments based on topics...")
    segments = []
    current_segment = {"start": 0, "end": 0, "content": "", "topic": None}

    for sentence in progressbar.progressbar(transcript):
        subjects = preprocess_text(sentence["content"])
        if not subjects:
            continue

        bow = dictionary.doc2bow([token for subject in subjects for token in subject])
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
    lda_model, corpus, dictionary = perform_topic_modeling(preprocessed_subjects, num_topics)
    
    save_checkpoint(project_path, CHECKPOINTS['TOPIC_MODELING_COMPLETE'], {
        'lda_model': lda_model,
        'corpus': corpus,
        'dictionary': dictionary
    })

    segments = identify_segments(transcript, lda_model, dictionary, num_topics)
    save_checkpoint(project_path, CHECKPOINTS['SEGMENTS_IDENTIFIED'], {
        'segments': segments
    })

    metadata = generate_metadata(segments, lda_model)

    results = {
        "topics": [
            {
                "topic_id": topic_id,
                "words": [word for word, _ in lda_model.show_topic(topic_id, topn=10)],
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