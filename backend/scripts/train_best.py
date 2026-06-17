"""
TrueScan - Best Model Training Pipeline v4
==========================================
Uses streaming + Parquet-native HuggingFace datasets for fast download.
Targets 97%+ accuracy with a 30-feature linguistic ensemble.

Run: python scripts/train_best.py
"""

import os, sys, re, json, time, pickle, math, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from scipy.sparse import hstack, csr_matrix

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] xgboost not installed — pip install xgboost")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "text_best")
MAX_PER_SOURCE = 12000
MIN_LEN = 80
MAX_LEN = 3000
SEED = 42
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Linguistic features (30) ─────────────────────────────────────────────────

AI_TRANSITIONS = {
    'furthermore','moreover','additionally','nevertheless','consequently',
    'therefore','however','thus','hence','specifically','importantly',
    'notably','significantly','essentially','ultimately','overall',
    'crucially','certainly','undoubtedly','arguably','indeed','accordingly',
    'subsequently','nonetheless','meanwhile','alternatively','conversely',
    'interestingly','remarkably','critically','particularly','generally'
}
AI_PHRASES = [
    'it is important to note','it is worth noting','it is crucial',
    'in conclusion','in summary','to summarize','to conclude',
    'as mentioned','as discussed','in other words','that being said',
    'first and foremost','last but not least','plays a crucial role',
    'in the realm of','it is essential','one of the most',
]
GPT_BUZZWORDS = re.compile(
    r'\b(?:delve|tapestry|nuanced|multifaceted|leverage|utilize|empower|foster|'
    r'pivotal|robust|synergy|paradigm|holistic|seamless|transformative|'
    r'groundbreaking|revolutionary|innovative|comprehensive|cutting.edge)\b'
)
FUNC_WORDS = {
    'the','a','an','in','on','at','to','of','is','are','was','were','be',
    'been','being','have','has','had','do','does','did','will','would',
    'could','should','may','might','shall','can','and','but','or','nor','so','yet'
}

def extract_features(texts):
    feats = []
    for text in texts:
        tl = text.lower()
        w  = re.findall(r'\b[a-zA-Z]{2,}\b', tl)
        s  = [x.strip() for x in re.split(r'[.!?]+', text) if len(x.strip()) > 5]
        wc = max(len(w), 1)
        sc = max(len(s), 1)
        sl = [len(re.findall(r'\b\w+\b', x)) for x in s]
        bg = [(w[i],w[i+1]) for i in range(len(w)-1)]
        tg = [(w[i],w[i+1],w[i+2]) for i in range(len(w)-2)]
        ln = text.split('\n')
        fw = w[:100]

        feats.append([
            len(set(w))/wc,                                                       # 1  TTR
            wc/sc,                                                                  # 2  avg sent len
            (np.std(sl)/max(np.mean(sl),1)) if len(sl)>=2 else 0,                # 3  burstiness
            1-(len(set(bg))/max(len(bg),1)),                                       # 4  bigram rep
            1-(len(set(tg))/max(len(tg),1)),                                       # 5  trigram rep
            sum(1 for x in w if len(x)>=7)/wc,                                    # 6  long word ratio
            sum(1 for c in text if c in '.,;:!?—–-()[]"\'')/max(len(text),1),    # 7  punct density
            sum(1 for c in text if c.isupper())/max(len(text),1),                 # 8  upper ratio
            sum(1 for c in text if c.isdigit())/max(len(text),1),                 # 9  digit ratio
            sum(len(x) for x in w)/wc,                                             # 10 avg word len
            len(set(x.split()[0].lower() if x.split() else '' for x in s))/sc,   # 11 starter div
            sum(1 for x in w if x in FUNC_WORDS)/wc,                              # 12 func ratio
            sum(1 for x in w if x in AI_TRANSITIONS)/wc,                          # 13 AI transition ratio
            math.log(max(len(text),1))/10,                                         # 14 log length
            len([p for p in text.split('\n\n') if p.strip()])/10,                 # 15 para count
            sum(1 for p in AI_PHRASES if p in tl)/len(AI_PHRASES),               # 16 AI phrase density
            text.count('?')/sc,                                                    # 17 question ratio
            text.count('!')/sc,                                                    # 18 exclaim ratio
            text.count(',')/max(len(text),1),                                      # 19 comma density
            (text.count('"')+text.count('\u201c')+text.count('\u201d'))/max(len(text),1),  # 20 quote ratio
            sum(1 for l in ln if l.strip().endswith(':') or l.strip().startswith('#'))/max(len(ln),1),  # 21 heading
            sum(1 for l in ln if re.match(r'^\s*[-*\u2022\d+\.]', l))/max(len(ln),1),  # 22 bullet
            max(sl)/50 if sl else 0,                                               # 23 max sent len
            min(sl)/10 if sl else 0,                                               # 24 min sent len
            len(set(fw))/max(len(fw),1),                                           # 25 first-100 TTR
            len(GPT_BUZZWORDS.findall(tl))/wc,                                    # 26 GPT buzzwords
            text.count('**')/max(len(text),1),                                    # 27 markdown bold
            text.count('- ')/max(len(ln),1),                                      # 28 dash list density
            len(re.findall(r'\b(?:firstly|secondly|thirdly|finally|lastly)\b', tl))/wc,  # 29 ordinal markers
            len(re.findall(r'[;:]', text))/max(len(text),1),                      # 30 colon/semicolon
        ])
    return np.array(feats, dtype=np.float32)


# ─── Data loaders (streaming where possible) ──────────────────────────────────

def balance(df, n):
    if df.empty: return df
    ai    = df[df.label==1].sample(min(n,(df.label==1).sum()), random_state=SEED)
    human = df[df.label==0].sample(min(n,(df.label==0).sum()), random_state=SEED)
    return pd.concat([ai,human]).sample(frac=1,random_state=SEED).reset_index(drop=True)

def clean(df):
    if df.empty: return df
    df = df.dropna(subset=['text','label'])
    df['text']  = df['text'].astype(str).str[:MAX_LEN]
    df['label'] = df['label'].astype(int)
    return df[df['text'].str.len() >= MIN_LEN].reset_index(drop=True)


def stream_hf(name, cfg, split, collect_fn, tag):
    """Stream a HuggingFace dataset and collect rows via collect_fn(item)->dict|None."""
    try:
        from datasets import load_dataset
        print(f"  [{tag}] Streaming {name}...")
        ds = load_dataset(name, cfg, split=split, streaming=True) if cfg \
             else load_dataset(name, split=split, streaming=True)
        rows = []
        for item in ds:
            r = collect_fn(item)
            if r:
                if isinstance(r, list): rows.extend(r)
                else: rows.append(r)
            if len(rows) >= MAX_PER_SOURCE * 3: break
        df = clean(pd.DataFrame(rows))
        df = balance(df, MAX_PER_SOURCE)
        print(f"    -> {len(df)} samples | AI:{(df.label==1).sum()} Human:{(df.label==0).sum()}")
        return df
    except Exception as e:
        print(f"    [SKIP {tag}] {e}")
        return pd.DataFrame()


def load_hc3():
    def collect(item):
        rows = []
        for t in (item.get('human_answers') or []):
            if t and len(str(t)) >= MIN_LEN:
                rows.append({'text': str(t)[:MAX_LEN], 'label': 0})
        for t in (item.get('chatgpt_answers') or []):
            if t and len(str(t)) >= MIN_LEN:
                rows.append({'text': str(t)[:MAX_LEN], 'label': 1})
        return rows
    return stream_hf("Hello-SimpleAI/HC3", "en", "train", collect, "HC3")


def load_gptwiki():
    def collect(item):
        rows = []
        h = item.get('wiki_intro','')
        a = item.get('generated_intro','')
        if h and len(h) >= MIN_LEN: rows.append({'text': h[:MAX_LEN], 'label': 0})
        if a and len(a) >= MIN_LEN: rows.append({'text': a[:MAX_LEN], 'label': 1})
        return rows
    return stream_hf("aadityaubhat/GPT-wiki-intro", None, "train", collect, "GPT-Wiki")


def load_m4():
    def collect(item):
        rows = []
        h = item.get('real_abstract','')
        a = item.get('generated_abstract','')
        if h and len(h) >= MIN_LEN: rows.append({'text': h[:MAX_LEN], 'label': 0})
        if a and len(a) >= MIN_LEN: rows.append({'text': a[:MAX_LEN], 'label': 1})
        return rows
    return stream_hf("NicolaiSivesind/ChatGPT-Research-Abstracts", None, "train", collect, "M4")


def load_llm_generated():
    def collect(item):
        t = str(item.get('text',''))
        l = item.get('generated', -1)
        if len(t) < MIN_LEN or l == -1: return None
        return {'text': t[:MAX_LEN], 'label': int(l)}
    return stream_hf("sudhanvasp/llm-generated-text-detection", None, "train", collect, "LLM-Gen")


def load_raid():
    def collect(item):
        t   = item.get('generation', item.get('text',''))
        src = str(item.get('model', item.get('source',''))).lower()
        if not t or len(t) < MIN_LEN: return None
        l = 0 if src in ('human','humans','') else 1
        return {'text': t[:MAX_LEN], 'label': l}
    return stream_hf("raid-bench/raid", None, "train", collect, "RAID")


def load_ai_writer():
    def collect(item):
        t = str(item.get('text',''))
        src = str(item.get('source','')).lower()
        if len(t) < MIN_LEN: return None
        l = 0 if any(x in src for x in ('human','pile','wikipedia','reddit','news')) else 1
        return {'text': t[:MAX_LEN], 'label': l}
    return stream_hf("artem9k/ai-text-detection-pile", None, "train", collect, "AI-Pile")


def load_daigt():
    """DAIGT V4 - the Kaggle-winning dataset, also on HuggingFace."""
    def collect(item):
        t = str(item.get('text', item.get('essay_text', '')))
        l = item.get('generated', item.get('label', -1))
        if len(t) < MIN_LEN or l == -1: return None
        return {'text': t[:MAX_LEN], 'label': int(l)}
    return stream_hf("dmitsab/daigt-v4-train-dataset", None, "train", collect, "DAIGT-V4")


def load_local():
    dfs = []
    base = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'text')
    if not os.path.isdir(base):
        return pd.DataFrame()
    for root, _, files in os.walk(base):
        for f in files:
            if not f.endswith('.csv'): continue
            try:
                df = pd.read_csv(os.path.join(root, f))
                if 'generated' in df.columns: df['label'] = df['generated'].astype(int)
                if 'label' not in df.columns: continue
                for col in ['essay_text','content','prompt','response','full_text']:
                    if col in df.columns and 'text' not in df.columns:
                        df = df.rename(columns={col:'text'}); break
                if 'text' not in df.columns: continue
                df = clean(df[['text','label']])
                df = balance(df, MAX_PER_SOURCE)
                if not df.empty:
                    print(f"    [LOCAL] {f}: {len(df)} samples")
                    dfs.append(df)
            except Exception as e:
                print(f"    [LOCAL skip] {f}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# ─── Training ─────────────────────────────────────────────────────────────────

def train():
    t0 = time.time()
    print("\n" + "="*65)
    print("  TrueScan Best Model Training v4")
    print("="*65)
    print(f"  Output: {OUTPUT_DIR}\n")

    loaders = [
        load_local,
        load_hc3, load_gptwiki, load_m4,
        load_llm_generated, load_raid,
        load_daigt, load_ai_writer,
    ]

    dfs = []
    for loader in loaders:
        df = loader()
        if not df.empty:
            dfs.append(df)

    if not dfs:
        print("[ERROR] No data loaded. Check internet connection.")
        sys.exit(1)

    df_all = pd.concat(dfs, ignore_index=True)
    df_all = clean(df_all)
    df_all = balance(df_all, 50000)
    print(f"\n  TOTAL: {len(df_all)} | AI:{(df_all.label==1).sum()} Human:{(df_all.label==0).sum()}")

    X_tr, X_vl, y_tr, y_vl = train_test_split(
        df_all['text'].tolist(), df_all['label'].tolist(),
        test_size=0.12, stratify=df_all['label'], random_state=SEED
    )
    y_tr = np.array(y_tr)
    y_vl = np.array(y_vl)
    print(f"  Train:{len(X_tr)}  Val:{len(X_vl)}\n")

    # TF-IDF
    print("  Fitting TF-IDF vectorizers...")
    tfidf_w = TfidfVectorizer(
        analyzer='word', ngram_range=(1,3), max_features=150000,
        sublinear_tf=True, min_df=2, strip_accents='unicode', dtype=np.float32
    )
    tfidf_c = TfidfVectorizer(
        analyzer='char_wb', ngram_range=(3,6), max_features=120000,
        sublinear_tf=True, min_df=3, dtype=np.float32
    )
    X_tr_w = tfidf_w.fit_transform(X_tr); X_vl_w = tfidf_w.transform(X_vl)
    X_tr_c = tfidf_c.fit_transform(X_tr); X_vl_c = tfidf_c.transform(X_vl)

    # Linguistic features
    print("  Extracting 30 linguistic features...")
    X_tr_l = extract_features(X_tr); X_vl_l = extract_features(X_vl)
    scaler = StandardScaler()
    X_tr_ls = scaler.fit_transform(X_tr_l)
    X_vl_ls = scaler.transform(X_vl_l)

    X_tr_full = hstack([X_tr_w, X_tr_c, csr_matrix(X_tr_ls)])
    X_vl_full = hstack([X_vl_w, X_vl_c, csr_matrix(X_vl_ls)])

    print("\n  Training classifiers...\n")

    # 1. Logistic Regression
    print("  [1/4] Logistic Regression...")
    lr = LogisticRegression(C=5.0, solver='saga', max_iter=1000,
                             random_state=SEED, class_weight='balanced')
    lr.fit(X_tr_full, y_tr)
    lr_p = lr.predict_proba(X_vl_full)[:,1]
    print(f"        Acc:{accuracy_score(y_vl,(lr_p>=0.5).astype(int)):.4f}  AUC:{roc_auc_score(y_vl,lr_p):.4f}")

    # 2. SGD (calibrated)
    print("  [2/4] SGD (calibrated)...")
    sgd_base = SGDClassifier(loss='modified_huber', alpha=1e-5, max_iter=500,
                              random_state=SEED, class_weight='balanced', n_jobs=-1)
    sgd_cal  = CalibratedClassifierCV(sgd_base, cv=3, method='isotonic')
    sgd_cal.fit(X_tr_full, y_tr)
    sgd_p = sgd_cal.predict_proba(X_vl_full)[:,1]
    print(f"        Acc:{accuracy_score(y_vl,(sgd_p>=0.5).astype(int)):.4f}  AUC:{roc_auc_score(y_vl,sgd_p):.4f}")

    # 3. XGBoost / GradientBoosting on linguistic features
    print("  [3/4] XGBoost on linguistic features...")
    if HAS_XGB:
        gb = XGBClassifier(
            n_estimators=800, learning_rate=0.03, max_depth=7,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.0,
            eval_metric='logloss', random_state=SEED, n_jobs=-1, verbosity=0
        )
    else:
        gb = GradientBoostingClassifier(
            n_estimators=400, learning_rate=0.06, max_depth=6, random_state=SEED
        )
    gb.fit(X_tr_ls, y_tr)
    gb_p = gb.predict_proba(X_vl_ls)[:,1]
    print(f"        Acc:{accuracy_score(y_vl,(gb_p>=0.5).astype(int)):.4f}  AUC:{roc_auc_score(y_vl,gb_p):.4f}")

    # 4. Random Forest on linguistic features
    print("  [4/4] Random Forest on linguistic features...")
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=12, min_samples_leaf=2,
        n_jobs=-1, random_state=SEED, class_weight='balanced'
    )
    rf.fit(X_tr_ls, y_tr)
    rf_p = rf.predict_proba(X_vl_ls)[:,1]
    print(f"        Acc:{accuracy_score(y_vl,(rf_p>=0.5).astype(int)):.4f}  AUC:{roc_auc_score(y_vl,rf_p):.4f}")

    # Grid-search optimal weights
    print("\n  Optimizing ensemble weights...")
    best_auc, best_w = 0, (0.5, 0.25, 0.15, 0.10)
    for w1 in np.arange(0.30, 0.75, 0.05):
        for w2 in np.arange(0.10, 0.45, 0.05):
            for w3 in np.arange(0.05, 0.35, 0.05):
                w4 = round(1.0 - w1 - w2 - w3, 4)
                if w4 < 0: continue
                ens = w1*lr_p + w2*sgd_p + w3*gb_p + w4*rf_p
                auc = roc_auc_score(y_vl, ens)
                if auc > best_auc:
                    best_auc = auc
                    best_w = (round(w1,2), round(w2,2), round(w3,2), round(w4,2))

    w_lr, w_sgd, w_gb, w_rf = best_w
    ens_p   = w_lr*lr_p + w_sgd*sgd_p + w_gb*gb_p + w_rf*rf_p
    ens_acc = accuracy_score(y_vl, (ens_p>=0.5).astype(int))
    ens_auc = roc_auc_score(y_vl, ens_p)

    print(f"\n  Best weights: LR={w_lr} SGD={w_sgd} GB={w_gb} RF={w_rf}")
    print(f"  Ensemble -> Acc:{ens_acc:.4f}  AUC:{ens_auc:.4f}\n")
    print(classification_report(y_vl, (ens_p>=0.5).astype(int), target_names=['Human','AI']))

    # Save all models
    print(f"\n  Saving to {OUTPUT_DIR}...")
    def save(obj, name):
        with open(os.path.join(OUTPUT_DIR, name), 'wb') as f:
            pickle.dump(obj, f)

    save(tfidf_w,  'tfidf_word.pkl')
    save(tfidf_c,  'tfidf_char.pkl')
    save(scaler,   'ling_scaler.pkl')
    save(lr,       'lr_model.pkl')
    save(sgd_cal,  'sgd_model.pkl')
    save(gb,       'gb_model.pkl')
    save(rf,       'rf_model.pkl')

    metrics = {
        "train_samples":     len(X_tr),
        "val_samples":       len(X_vl),
        "lr_val_acc":        round(float(accuracy_score(y_vl,(lr_p>=0.5).astype(int))), 4),
        "lr_val_auc":        round(float(roc_auc_score(y_vl,lr_p)), 4),
        "sgd_val_acc":       round(float(accuracy_score(y_vl,(sgd_p>=0.5).astype(int))), 4),
        "sgd_val_auc":       round(float(roc_auc_score(y_vl,sgd_p)), 4),
        "gb_val_acc":        round(float(accuracy_score(y_vl,(gb_p>=0.5).astype(int))), 4),
        "gb_val_auc":        round(float(roc_auc_score(y_vl,gb_p)), 4),
        "rf_val_acc":        round(float(accuracy_score(y_vl,(rf_p>=0.5).astype(int))), 4),
        "rf_val_auc":        round(float(roc_auc_score(y_vl,rf_p)), 4),
        "ensemble_val_acc":  round(float(ens_acc), 4),
        "ensemble_val_auc":  round(float(ens_auc), 4),
        "training_time_minutes": round((time.time()-t0)/60, 2),
        "ensemble_weights":  {"lr": w_lr, "sgd": w_sgd, "gb": w_gb, "rf": w_rf},
        "n_ling_features":   30,
        "has_xgboost":       HAS_XGB,
        "datasets_loaded":   len(dfs),
    }
    with open(os.path.join(OUTPUT_DIR, 'training_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    elapsed = (time.time()-t0)/60
    print(f"\n{'='*65}")
    print(f"  DONE in {elapsed:.1f} min")
    print(f"  Ensemble Accuracy : {ens_acc*100:.2f}%")
    print(f"  Ensemble AUC      : {ens_auc:.4f}")
    print(f"  Models saved to   : {OUTPUT_DIR}")
    print(f"{'='*65}\n")
    return ens_acc


if __name__ == '__main__':
    acc = train()
    if acc >= 0.97:
        print("[EXCELLENT] >= 97% accuracy!")
    elif acc >= 0.95:
        print("[GOOD] >= 95% accuracy")
    else:
        print("[OK] Training complete — add more data for better results")
