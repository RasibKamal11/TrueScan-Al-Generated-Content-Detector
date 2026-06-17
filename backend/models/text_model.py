from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
import os
import re
import math
import numpy as np
import json
from loguru import logger
import asyncio
from .external_api import get_external_integrator

class TextDetector:
    def __init__(self):
        logger.info("Loading Simplified Text Detection Model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._neural_loaded = False
        
        # Load HuggingFace neural classifier (Hello-SimpleAI/chatgpt-detector-roberta)
        for model_id in [
            "Hello-SimpleAI/chatgpt-detector-roberta",
            "roberta-base-openai-detector",
        ]:
            try:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
                    self.model = AutoModelForSequenceClassification.from_pretrained(model_id, local_files_only=True).to(self.device)
                except Exception:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_id)
                    self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(self.device)
                
                self.model.eval()
                self._neural_loaded = True
                logger.success(f"Neural TextDetector loaded: {model_id}")
                break
            except Exception as e:
                logger.warning(f"Failed loading {model_id}: {e}")

    def _neural_predict(self, text: str) -> float:
        if not self._neural_loaded:
            return 0.5
        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
            # Index 1 corresponds to AI/ChatGPT class
            return float(probs[0][1].item())
        except Exception:
            return 0.5

    def predict(self, text: str) -> float:
        """Main prediction. Returns 0.0 (Human) to 1.0 (AI)."""
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 5]
        
        if not words or not sentences:
            return 0.5

        # 1. Perplexity (entropy of word frequencies)
        word_counts = {}
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1
        total_words = len(words)
        probs = [count / total_words for count in word_counts.values()]
        entropy = -sum(p * math.log2(p) for p in probs)
        perplexity = 2 ** entropy
        perplexity_score = min(100.0, max(0.0, (perplexity / 120.0) * 100))

        # 2. Vocabulary diversity (TTR)
        ttr = len(set(words)) / total_words
        vocab_score = min(100.0, max(0.0, ttr * 100))

        # 3. Sentence variation (standard deviation of sentence lengths)
        sentence_lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
        sentence_var = float(np.std(sentence_lengths)) if len(sentence_lengths) >= 2 else 0.0
        sent_var_score = min(100.0, (sentence_var / 15.0) * 100)

        # 4. Repetition detection (bigram/trigram repetition rate)
        bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
        trigrams = [(words[i], words[i+1], words[i+2]) for i in range(len(words)-2)]
        bigram_rep = 1.0 - (len(set(bigrams)) / len(bigrams)) if bigrams else 0.0
        trigram_rep = 1.0 - (len(set(trigrams)) / len(trigrams)) if trigrams else 0.0
        repetition_score = min(100.0, max(0.0, (0.6 * bigram_rep + 0.4 * trigram_rep) * 100))

        # Normalize features
        p_perp = 1.0 - (perplexity_score / 100.0)
        p_vocab = 1.0 - (vocab_score / 100.0)
        p_sent = 1.0 - (sent_var_score / 100.0)
        p_rep = repetition_score / 100.0

        feature_score = 0.25 * p_perp + 0.25 * p_vocab + 0.25 * p_sent + 0.25 * p_rep
        feature_score = float(np.clip(feature_score, 0.02, 0.98))

        # Blend neural and feature predictions
        neural_score = self._neural_predict(text)
        if self._neural_loaded:
            blended = 0.50 * neural_score + 0.50 * feature_score
        else:
            blended = feature_score

        return float(np.clip(blended, 0.0, 1.0))

    def _generate_explanation(self, score: float, confidence: float, metrics: dict, classification: str) -> str:
        v_rich = metrics["vocabulary_richness"]
        if v_rich > 70:
            v_desc = "high vocabulary diversity"
        elif v_rich < 40:
            v_desc = "limited vocabulary diversity"
        else:
            v_desc = "moderate vocabulary diversity"

        s_var = metrics["sentence_length_variance"]
        if s_var > 30:
            s_desc = "natural sentence variation"
        else:
            s_desc = "uniform sentence structure"

        r_score = metrics["repetition_score"]
        if r_score > 15:
            r_desc = "Some repetitive patterns were detected"
        else:
            r_desc = "few repetitive patterns were observed"

        return f"The text shows {v_desc} and {s_desc}. {r_desc}, resulting in a {classification} classification."

    def predict_detailed(self, text: str) -> dict:
        """Full analysis with per-sentence scores and core linguistic metrics."""
        raw_chunks = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'"\(])', text)
        chunks = [c.strip() for c in raw_chunks if c.strip()]
        
        overall_score = self.predict(text)
        
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        total_words = max(len(words), 1)
        
        # Calculate features for agreement consensus
        word_counts = {}
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1
        probs = [count / total_words for count in word_counts.values()]
        entropy = -sum(p * math.log2(p) for p in probs)
        perplexity = 2 ** entropy
        perplexity_score = min(100.0, max(0.0, (perplexity / 120.0) * 100))
        ttr = len(set(words)) / total_words
        vocab_score = min(100.0, max(0.0, ttr * 100))
        
        sentence_lengths = [len(re.findall(r'\b\w+\b', s)) for s in chunks if len(s) >= 10]
        if not sentence_lengths:
            sentence_lengths = [len(text.split())]
        sentence_var = float(np.std(sentence_lengths)) if len(sentence_lengths) >= 2 else 0.0
        sent_var_score = min(100.0, (sentence_var / 15.0) * 100)
        
        bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
        trigrams = [(words[i], words[i+1], words[i+2]) for i in range(len(words)-2)]
        bigram_rep = 1.0 - (len(set(bigrams)) / len(bigrams)) if bigrams else 0.0
        trigram_rep = 1.0 - (len(set(trigrams)) / len(trigrams)) if trigrams else 0.0
        repetition_score = min(100.0, max(0.0, (0.6 * bigram_rep + 0.4 * trigram_rep) * 100))

        p_perp = 1.0 - (perplexity_score / 100.0)
        p_vocab = 1.0 - (vocab_score / 100.0)
        p_sent = 1.0 - (sent_var_score / 100.0)
        p_rep = repetition_score / 100.0

        neural_score = self._neural_predict(text)
        is_ai = overall_score > 0.5
        
        indicators = []
        if self._neural_loaded:
            indicators.append(neural_score > 0.5)
        indicators.append(p_perp > 0.5)
        indicators.append(p_vocab > 0.5)
        indicators.append(p_sent > 0.5)
        indicators.append(p_rep > 0.5)

        agreement = sum(1 for ind in indicators if ind == is_ai)
        confidence_score = float(np.clip(agreement / len(indicators), 0.1, 1.0))

        # Classification mapping
        if confidence_score < 0.4:
            classification = "Uncertain"
        elif 0.4 <= overall_score <= 0.6:
            classification = "Mixed"
        elif overall_score > 0.6:
            classification = "Likely AI"
        else:
            classification = "Likely Human"

        metrics_dict = {
            "perplexity": round(perplexity_score, 1),
            "burstiness": round(sent_var_score, 1),
            "vocabulary_richness": round(vocab_score, 1),
            "neural_repetition": round(overall_score * 100, 1),
            "repetition_score": round(repetition_score, 1),
            "entropy": round(min(100.0, (entropy / 8.0) * 100), 1),
            "readability": round(min(100.0, (ttr * 80)), 1),
            "sentence_length_variance": round(float(np.var(sentence_lengths)), 1),
            "stylometric_score": round(vocab_score, 1),
            "semantic_coherence": round((1.0 - bigram_rep) * 100, 1),
            "sentence_count": len(chunks),
            "avg_sentence_length": round(sum(sentence_lengths)/max(len(sentence_lengths), 1), 1),
            "word_count": total_words,
            "model_type": "neural" if self._neural_loaded else "fallback"
        }

        explanation = self._generate_explanation(overall_score, confidence_score, metrics_dict, classification)

        # Build sentence list for highlights
        sentences_list = []
        for idx, chunk in enumerate(chunks):
            if len(chunk) < 10:
                continue
            chunk_words = re.findall(r'\b[a-zA-Z]{2,}\b', chunk.lower())
            chunk_ttr = len(set(chunk_words)) / max(len(chunk_words), 1) if chunk_words else 0.5
            chunk_score = 0.5 * overall_score + 0.5 * (1.0 - chunk_ttr)
            
            sentences_list.append({
                "text": chunk,
                "ai_probability": round(float(np.clip(chunk_score, 0.0, 1.0)), 4),
                "word_count": len(chunk.split())
            })

        if not sentences_list:
            sentences_list = [{
                "text": text,
                "ai_probability": round(overall_score, 4),
                "word_count": len(text.split())
            }]

        return {
            "score": round(overall_score, 4),
            "ai_probability": round(overall_score, 4),
            "human_probability": round(1.0 - overall_score, 4),
            "confidence_score": round(confidence_score, 4),
            "classification": classification,
            "predicted_source": "Likely AI" if overall_score > 0.5 else "Likely Human",
            "explanation": explanation,
            "sentences": sentences_list,
            "metrics": metrics_dict
        }

    async def predict_deep(self, text: str) -> dict:
        """Call external integrator in parallel if available, otherwise fallback gracefully."""
        base_result = self.predict_detailed(text)
        try:
            integrator = get_external_integrator()
            gemini_res = await integrator.get_gemini_analysis(text)
            if gemini_res and "ai_score" in gemini_res:
                final_score = 0.5 * base_result["score"] + 0.5 * gemini_res["ai_score"]
                base_result["score"] = round(final_score, 4)
                base_result["ai_probability"] = round(final_score, 4)
                base_result["human_probability"] = round(1.0 - final_score, 4)
                base_result["predicted_source"] = gemini_res.get("source", base_result["predicted_source"])
        except Exception:
            pass
        return base_result
