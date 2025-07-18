#!/usr/bin/env python3
"""
Test script for enhanced transcript analysis with spaCy.
This script tests the new spaCy-powered transcript analysis functionality.
"""

import sys
import os

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from video_topic_splitter.analysis.enhanced_transcript_analysis import analyze_transcript_with_spacy

def test_enhanced_transcript_analysis():
    """Test the enhanced transcript analysis with sample data."""
    
    # Sample transcript segment (mimicking real transcript data)
    sample_transcript = [
        {
            "start": 0.0,
            "end": 3.5,
            "content": "Good morning everyone. Let's discuss Project Phoenix."
        },
        {
            "start": 3.5,
            "end": 8.2,
            "content": "Susan Miller reported that the integration with the Titan API is facing delays."
        },
        {
            "start": 8.2,
            "end": 12.8,
            "content": "We need to resolve this by next Friday, June 21st."
        },
        {
            "start": 12.8,
            "end": 18.1,
            "content": "The client, Acme Corporation, is expecting a demo with the new user authentication feature."
        },
        {
            "start": 18.1,
            "end": 22.5,
            "content": "The revised budget for this phase is approximately $55,000."
        },
        {
            "start": 22.5,
            "end": 28.0,
            "content": "John Davis from our London office will help the development team debug the API integration."
        }
    ]
    
    print("🧪 Testing Enhanced Transcript Analysis with spaCy")
    print("=" * 60)
    
    try:
        # Perform the analysis
        results = analyze_transcript_with_spacy(sample_transcript)
        
        if "error" in results:
            print(f"❌ Analysis failed: {results['error']}")
            return False
        
        print("✅ Analysis completed successfully!")
        print()
        
        # Display basic metrics
        basic_metrics = results.get("basic_metrics", {})
        print(f"📊 Basic Metrics:")
        print(f"  • Token count: {basic_metrics.get('token_count', 0)}")
        print(f"  • Sentence count: {basic_metrics.get('sentence_count', 0)}")
        print(f"  • Duration: {basic_metrics.get('duration', 0):.1f} seconds")
        print(f"  • Speech rate: {basic_metrics.get('speech_rate', 0):.2f} tokens/second")
        print()
        
        # Display key phrases
        key_phrases = results.get("key_phrases", {})
        print(f"🔑 Key Phrases:")
        noun_phrases = key_phrases.get("noun_phrases", {})
        print(f"  • Noun phrases: {list(noun_phrases.keys())[:5]}")
        key_lemmas = key_phrases.get("key_lemmas", {})
        print(f"  • Key lemmas: {list(key_lemmas.keys())[:5]}")
        technical_terms = key_phrases.get("technical_terms", {})
        print(f"  • Technical terms: {list(technical_terms.keys())[:5]}")
        print()
        
        # Display named entities
        named_entities = results.get("named_entities", {})
        entities = named_entities.get("entities", {})
        print(f"🏷️  Named Entities:")
        for entity_type, entity_list in entities.items():
            if entity_list:
                entity_texts = [e["text"] if isinstance(e, dict) else e for e in entity_list]
                print(f"  • {entity_type}: {entity_texts}")
        print()
        
        # Display actions and relationships
        actions = results.get("actions_and_relationships", {})
        svo_triplets = actions.get("subject_verb_object_triplets", [])
        print(f"🎯 Actions & Relationships:")
        if svo_triplets:
            for i, triplet in enumerate(svo_triplets[:3]):
                verb = triplet.get("verb", "unknown")
                subjects = triplet.get("subjects", [])
                objects = triplet.get("objects", [])
                print(f"  • Action {i+1}: {subjects} → {verb} → {objects}")
        else:
            print("  • No subject-verb-object relationships found")
        print()
        
        # Display technical elements
        technical_elements = results.get("technical_elements", [])
        print(f"⚙️  Technical Elements:")
        if technical_elements:
            print(f"  • {', '.join(technical_elements[:10])}")
        else:
            print("  • No technical elements detected")
        print()
        
        # Display semantic features (if available)
        semantic_features = results.get("semantic_features", {})
        if semantic_features and not semantic_features.get("error"):
            print(f"🧠 Semantic Features:")
            print(f"  • Document coherence: {semantic_features.get('document_coherence', 0):.3f}")
            print(f"  • Vector analysis available: {semantic_features.get('semantic_analysis_available', False)}")
        else:
            print(f"🧠 Semantic Features: Not available (requires en_core_web_md model)")
        print()
        
        print("✅ Enhanced transcript analysis test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_with_basic_analysis():
    """Compare enhanced spaCy analysis with basic string-based analysis."""
    
    sample_text = "The development team will analyze the user authentication system. We're analyzing performance metrics and analyzing security protocols."
    
    print("\n🔍 Comparison: Basic vs Enhanced Analysis")
    print("=" * 60)
    
    # Basic analysis (old method)
    print("📝 Basic Analysis (old method):")
    words = sample_text.lower().split()
    word_freq = {}
    for word in words:
        if len(word) > 3:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    basic_phrases = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  • Top words: {[phrase[0] for phrase in basic_phrases]}")
    print()
    
    # Enhanced analysis
    print("🚀 Enhanced Analysis (spaCy):")
    try:
        sample_transcript = [{"start": 0, "end": 10, "content": sample_text}]
        results = analyze_transcript_with_spacy(sample_transcript)
        
        if "error" not in results:
            key_phrases = results.get("key_phrases", {})
            key_lemmas = key_phrases.get("key_lemmas", {})
            print(f"  • Key lemmas (normalized): {list(key_lemmas.keys())[:5]}")
            
            technical_terms = key_phrases.get("technical_terms", {})
            print(f"  • Technical terms: {list(technical_terms.keys())[:5]}")
            
            named_entities = results.get("named_entities", {})
            entities = named_entities.get("entities", {})
            print(f"  • Named entities: {dict(entities)}")
        else:
            print(f"  • Analysis failed: {results['error']}")
    
    except Exception as e:
        print(f"  • Enhanced analysis error: {e}")
    
    print("\n✨ Key Improvements:")
    print("  • Lemmatization groups 'analyze', 'analyzing', 'analyzed' together")
    print("  • Stop word removal eliminates 'the', 'will', 'and' etc.")
    print("  • Technical term detection identifies domain-specific vocabulary")
    print("  • Named entity recognition extracts structured information")

if __name__ == "__main__":
    print("🎬 Video Topic Splitter - Enhanced Transcript Analysis Test")
    print("=" * 60)
    
    # Test basic functionality
    success = test_enhanced_transcript_analysis()
    
    # Show comparison with basic analysis
    compare_with_basic_analysis()
    
    if success:
        print("\n🎉 All tests passed! Enhanced transcript analysis is ready to use.")
    else:
        print("\n⚠️  Some tests failed. Please check the installation and try again.")
        print("\n💡 Make sure to install spaCy and download the language model:")
        print("   pip install spacy")
        print("   python -m spacy download en_core_web_md")