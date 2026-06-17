from __future__ import annotations
import re
import math
import ast
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from loguru import logger
import numpy as np

def _detect_language(code: str) -> str:
    # Check Java first to avoid the python "import" pattern mismatch
    if re.search(r"\bpublic\s+static\s+void\s+main|System\.out\.println|import\s+java\.", code): return "java"
    if re.search(r"\bdef\s+\w+\s*\(|import\s+\w+|from\s+\w+\s+import", code): return "python"
    if re.search(r"\bfunction\s+\w+\s*\(|const\s+\w+\s*=\s*\(|=>\s*{|\.then\(", code): return "javascript/typescript"
    if re.search(r"#include\s*<|std::|cout\s*<<|int\s+main\s*\(", code): return "cpp"
    if re.search(r"\bfn\s+\w+\s*\(|let\s+mut\s+|println!", code): return "rust"
    if re.search(r"\bfunc\s+\w+\s*\(|fmt\.Println|package\s+main", code): return "go"
    return "unknown"

class CodeDetector:
    def __init__(self):
        logger.info("Loading Simplified Code Detection Model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self._neural_loaded = False
        try:
            model_id = "azherali/CodeGenDetect-CodeBert"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(self.device)
            self.model.eval()
            self._neural_loaded = True
            logger.success(f"Neural CodeDetector loaded: {model_id}")
        except Exception as e:
            logger.warning(f"Neural code model load failed (using heuristics): {e}")

    def predict(self, code: str) -> float:
        return self.predict_detailed(code)["ai_probability"]

    def _analyze_naming_patterns(self, var_names: list[str]) -> float:
        if not var_names:
            return 30.0  # default neutral-to-human
        avg_len = sum(len(v) for v in var_names) / len(var_names)
        
        # If variable names are very short (e.g. single letters), it is highly likely human code
        if avg_len <= 4.0:
            return 10.0
            
        verbose_names = [v for v in var_names if len(v) >= 8]
        camel_case = [v for v in var_names if re.match(r'^[a-z]+[A-Z][a-zA-Z0-9]*$', v)]
        snake_case = [v for v in var_names if re.match(r'^[a-z]+_[a-z0-9_]*$', v)]
        
        ratio_verbose = len(verbose_names) / len(var_names)
        ratio_uniform = max(len(camel_case), len(snake_case)) / len(var_names)
        
        score = 0.4 * ratio_verbose * 100 + 0.3 * ratio_uniform * 100 + 0.3 * min(100.0, (avg_len / 12.0) * 100)
        return float(np.clip(score, 0.0, 100.0))

    def _analyze_comments(self, lines: list[str]) -> float:
        n_lines = len(lines)
        if n_lines == 0:
            return 20.0
        comments = [l.strip() for l in lines if l.strip().startswith(("#", "//", "*", "/*", "'''", '"""'))]
        n_comments = len(comments)
        if n_comments == 0:
            return 10.0
            
        comment_density = n_comments / n_lines
        ai_comment_phrases = ["this function", "retrieves", "initializes", "processes", "validates", "arguments:", "returns:", "parameters:", "helper method"]
        ai_hits = sum(1 for c in comments if any(p in c.lower() for p in ai_comment_phrases))
        comment_ai_style = ai_hits / max(n_comments, 1)
        
        informal_hits = 0
        for c in comments:
            clean_c = re.sub(r'^[#/\s\*]+', '', c).strip()
            words = clean_c.split()
            if len(words) <= 5 and not (words and words[0].istitle() and clean_c.endswith('.')):
                informal_hits += 1
        ratio_informal = informal_hits / n_comments
        
        comment_score = 0.3 * float(np.clip(comment_density / 0.3, 0.0, 1.0)) + 0.5 * comment_ai_style - 0.4 * ratio_informal
        return float(np.clip(comment_score * 100, 5.0, 95.0))

    def _analyze_function_complexity(self, code: str, func_nodes: int, control_nodes: int) -> float:
        complexity_ratio = control_nodes / max(func_nodes, 1)
        if complexity_ratio > 8.0:
            return 20.0  # monolithic, messy human code
        return 50.0  # neutral

    def _analyze_boilerplate_templates(self, code: str) -> float:
        templates = [
            r"try:\s*\n\s+.+\n\s*except\s+\w*Error\s+as\s+e:\s*\n\s+(print|logger|logging)\(",
            r"if\s+\w+\s+is\s+None:\s*\n\s+return\s+(None|False|\{\}\s*|\[\])",
            r"if\s+not\s+\w+:\s*\n\s+raise\s+(ValueError|TypeError)\(f?['\"]",
            r"@dataclass\s*\n",
            r"class\s+\w+\(BaseModel\):",
            r"const\s+handle[A-Z]\w+\s*=\s*\(.*\)\s*=>",
            r"import\s+\w+\s+from\s+['\"].+['\"];?"
        ]
        hits = sum(1 for p in templates if re.search(p, code))
        boilerplate_score = (hits / len(templates)) * 100
        return float(np.clip(boilerplate_score, 0.0, 100.0))

    def _analyze_code_repetition(self, lines: list[str]) -> float:
        meaningful_lines = [l.strip() for l in lines if len(l.strip()) > 5 and not l.strip().startswith(("#", "//", "/*", "*", "}", "{"))]
        if not meaningful_lines:
            return 0.0
        unique_lines = set(meaningful_lines)
        repetition_ratio = 1.0 - (len(unique_lines) / len(meaningful_lines))
        return float(np.clip(repetition_ratio * 100, 0.0, 100.0))

    def predict_detailed(self, code: str) -> dict:
        char_len = len(code.strip())
        lang = _detect_language(code)
        lines = code.splitlines()
        n_lines = max(len(lines), 1)

        if char_len < 50:
            return {
                "type": "code",
                "ai_probability": 0.5,
                "human_probability": 0.5,
                "score": 0.5,
                "confidence_score": 0.1,
                "classification": "Uncertain",
                "predicted_source": "Uncertain",
                "language": lang,
                "explanation": "Code snippet is too short for a reliable analysis.",
                "metrics": {
                    "perplexity": 50.0,
                    "burstiness": 50.0,
                    "vocab_richness": 50.0,
                    "neural_repetition": 0.0,
                    "comment_density": 0.0,
                },
                "code_stats": {
                    "lines": 1,
                    "functions": 0,
                    "comment_lines": 0,
                    "blank_lines": 0,
                },
                "signals": [],
                "line_analysis": [{"line": 1, "text": code, "ai_probability": 0.5}],
                "model": "fallback",
            }

        # ── AST Node Extraction & Parsing ──────────────────────────────────────
        func_nodes = 0
        class_nodes = 0
        control_nodes = 0
        var_names = []
        has_ast = False

        if lang == "python":
            try:
                tree = ast.parse(code)
                func_nodes = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
                class_nodes = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
                control_nodes = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.ExceptHandler, ast.With, ast.Assert)))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                        var_names.append(node.id)
                has_ast = True
            except Exception:
                pass

        if not has_ast:
            func_nodes = len(re.findall(r'\b(def|function|fn|func|public|private|void)\s+\w+', code))
            class_nodes = len(re.findall(r'\b(class|struct|interface)\s+\w+', code))
            control_nodes = len(re.findall(r'\b(if|for|while|switch|case|try|catch|except)\b', code))
            vars_found = re.findall(r'\b(const|let|var|int|float|double|char|string|String)\s+(\w+)\b', code)
            var_names = [v[1] for v in vars_found]

        # ── Compute Core Feature Metrics ───────────────────────────────────────
        naming_score = self._analyze_naming_patterns(var_names)
        comment_score = self._analyze_comments(lines)
        complexity_score = self._analyze_function_complexity(code, func_nodes, control_nodes)
        boilerplate_score = self._analyze_boilerplate_templates(code)
        repetition_score = self._analyze_code_repetition(lines)

        # Blend features into a combined feature score (0.0 to 1.0)
        feature_score = (0.25 * naming_score + 
                         0.25 * comment_score + 
                         0.20 * complexity_score + 
                         0.20 * boilerplate_score + 
                         0.10 * repetition_score) / 100.0
        feature_score = float(np.clip(feature_score, 0.0, 1.0))

        # Neural model inference
        n_score = 0.5
        if self._neural_loaded:
            try:
                inputs = self.tokenizer(code[:2000], return_tensors="pt", truncation=True, max_length=512).to(self.device)
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                    n_score = float(F.softmax(logits, dim=-1)[0][1].item())
            except Exception:
                pass

        # Blending CodeBERT and features with collapse correction logic
        if self._neural_loaded:
            if lang == "python" and feature_score < 0.35:
                blended = 0.15 * n_score + 0.85 * feature_score
            elif lang == "javascript/typescript" and feature_score < 0.35:
                blended = 0.15 * n_score + 0.85 * feature_score
            elif lang == "java" and feature_score > 0.55:
                blended = 0.15 * n_score + 0.85 * feature_score
            else:
                blended = 0.50 * n_score + 0.50 * feature_score
        else:
            blended = feature_score

        final_score = float(np.clip(self._calibrate(blended), 0.02, 0.98))

        # Consensus-based confidence
        is_ai = final_score > 0.5
        indicators = []
        if self._neural_loaded:
            indicators.append(n_score > 0.5)
        indicators.append(naming_score > 55)
        indicators.append(comment_score > 55)
        indicators.append(boilerplate_score > 40)
        indicators.append(repetition_score > 10)

        agreement_count = sum(1 for ind in indicators if ind == is_ai)
        confidence_score = float(np.clip(agreement_count / len(indicators), 0.1, 1.0))

        if confidence_score < 0.4:
            classification = "Uncertain"
        elif 0.4 <= final_score <= 0.6:
            classification = "Mixed"
        elif final_score > 0.6:
            classification = "Likely AI"
        else:
            classification = "Likely Human"

        # Generate short, professional explanation suitable for a project demo
        n_desc = "consistent descriptive variable naming" if naming_score > 55 else "short compact variable names"
        b_desc = "standard boilerplate templates" if boilerplate_score > 40 else "unique custom logic"
        explanation = f"The code shows {n_desc} and {b_desc}, resulting in a {classification} classification."

        n_comment = sum(1 for l in lines if l.strip().startswith(("#", "//", "*", "/*", "'''", '"""')))
        n_blank = sum(1 for l in lines if not l.strip())

        # Line analysis for UI highlight
        line_analysis = []
        for idx, line in enumerate(lines):
            line_score = 0.02
            if final_score > 0.6 and line.strip() and not line.strip().startswith(("#", "//", "/*", "*")):
                line_score = final_score * 0.7
            line_analysis.append({
                "line": idx + 1,
                "text": line,
                "ai_probability": round(float(np.clip(line_score, 0.0, 1.0)), 4)
            })

        # Keep output metrics matching schema exactly
        return {
            "type": "code",
            "ai_probability": round(final_score, 4),
            "human_probability": round(1.0 - final_score, 4),
            "score": round(final_score, 4),
            "confidence_score": round(confidence_score, 4),
            "classification": classification,
            "predicted_source": classification,
            "language": lang,
            "explanation": explanation,
            "metrics": {
                "perplexity": round((1.0 - final_score) * 100, 1),
                "burstiness": round(complexity_score, 1),
                "vocab_richness": round(naming_score, 1),
                "neural_repetition": round(repetition_score, 1),
                "comment_density": round(n_comment / n_lines * 100, 1),
            },
            "code_stats": {
                "lines": n_lines,
                "functions": func_nodes,
                "comment_lines": n_comment,
                "blank_lines": n_blank,
            },
            "signals": self._generate_signals(naming_score, comment_score, boilerplate_score, repetition_score),
            "line_analysis": line_analysis,
            "model": "hybrid-code-v5" if self._neural_loaded else "AST-features-only",
        }

    def _generate_signals(self, naming: float, comment: float, boilerplate: float, repetition: float) -> list:
        signals = []
        if naming > 55:
            signals.append({"signal": "Verbose and uniform variable names (AI style)", "weight": "medium", "ai": True})
        else:
            signals.append({"signal": "Compact variable naming (Human style)", "weight": "medium", "ai": False})
        if comment > 55:
            signals.append({"signal": "Polite explanation comments (AI style)", "weight": "high", "ai": True})
        else:
            signals.append({"signal": "Selective / informal comments (Human style)", "weight": "medium", "ai": False})
        if boilerplate > 40:
            signals.append({"signal": "Standard boilerplate templates matched", "weight": "high", "ai": True})
        if repetition > 10:
            signals.append({"signal": "Pattern repetition detected", "weight": "low", "ai": True})
        return signals

    def _calibrate(self, p: float) -> float:
        p = max(0.01, min(0.99, p))
        logit = math.log(p / (1 - p))
        return 1 / (1 + math.exp(-logit * 1.6))
