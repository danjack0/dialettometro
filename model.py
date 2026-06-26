"""
Italian Dialect Classifier — neural network, trained from scratch
=================================================================
    python model.py --data balanced.csv --extra-train testset_train.csv --test testset_eval.csv
    python model.py ... --features both      # char + word features

--features char  (default) character n-grams only
           word               word n-grams only
           both               char + word combined (targets close-dialect confusion)
"""

import argparse, csv, copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

torch.manual_seed(0); np.random.seed(0)

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="balanced.csv")
ap.add_argument("--extra-train", default=None)
ap.add_argument("--test", default=None)
ap.add_argument("--features", default="char", choices=["char", "word", "both"])
ap.add_argument("--weight-standard", type=float, default=1.0,
                help="loss weight for the standard class; <1 makes the model predict it less eagerly")
ap.add_argument("--save", default=None,
                help="path to save the trained model + vectorizer bundle (e.g. dialect_model.pt)")
args = ap.parse_args()

# REGULARIZATION TWEAK: Increased WEIGHT_DECAY to 1e-3 to fix immediate 1.000 training accuracy saturation
EPOCHS, BATCH, HIDDEN, LR, WEIGHT_DECAY, PATIENCE = 200, 32, 128, 1e-3, 1e-3, 15


def read_csv(path):
    rows = list(csv.reader(open(path, encoding="utf-8")))
    has_src = len(rows[0]) >= 3
    return [(r[0], r[1], r[2] if has_src and len(r) >= 3 else "")
            for r in rows[1:] if len(r) >= 2]


def build_vectorizer(kind):
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=5000)
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=5000)
    if kind == "char": return char
    if kind == "word": return word
    return FeatureUnion([("char", char), ("word", word)])   # both


# 1. LOAD
data = read_csv(args.data)
sentences = np.array([d[0] for d in data], dtype=object)
labels = [d[1] for d in data]
sources = np.array([d[2] for d in data], dtype=object)
classes = sorted(set(labels)); class_to_id = {c: i for i, c in enumerate(classes)}
y = np.array([class_to_id[l] for l in labels])
print(f"Loaded {len(sentences)} sentences | classes: {classes} | features: {args.features}")

# 2. SPLIT (source-aware)
if any(s for s in sources) and len(set(sources)) > len(classes):
    from sklearn.model_selection import GroupShuffleSplit
    tr_idx, va_idx = next(GroupShuffleSplit(1, test_size=0.2, random_state=0)
                          .split(sentences, y, groups=sources))
    print(f">> Honest split: validating on UNSEEN articles ({len(set(sources[va_idx]))} held out).")
else:
    from sklearn.model_selection import train_test_split as _tts
    tr_idx, va_idx = _tts(np.arange(len(y)), test_size=0.2, stratify=y, random_state=0)

X_train_txt = list(sentences[tr_idx]); y_train = y[tr_idx]
X_val_txt = list(sentences[va_idx]);   y_val = y[va_idx]

# 2b. MIX casual training data
if args.extra_train:
    extra = read_csv(args.extra_train)
    ex_txt = [e[0] for e in extra if e[1] in class_to_id]
    ex_y = [class_to_id[e[1]] for e in extra if e[1] in class_to_id]
    X_train_txt += ex_txt
    y_train = np.concatenate([y_train, np.array(ex_y, dtype=y_train.dtype)])
    print(f">> Mixed {len(ex_txt)} casual sentences into TRAINING.")

# 3. FEATURES
vectorizer = build_vectorizer(args.features)
X_train = vectorizer.fit_transform(X_train_txt).toarray().astype(np.float32)
X_val = vectorizer.transform(X_val_txt).toarray().astype(np.float32)
input_dim = X_train.shape[1]
print(f"Feature vector size: {input_dim}")

X_train_t = torch.from_numpy(X_train); y_train_t = torch.from_numpy(y_train).long()
X_val_t = torch.from_numpy(X_val);     y_val_t = torch.from_numpy(y_val).long()


# 4. NETWORK
class DialectNet(nn.Module):
    def __init__(self, in_dim, hidden, n):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden); self.relu = nn.ReLU()
        # REGULARIZATION TWEAK: Increased dropout from 0.4 to 0.5 to reduce pattern memorization
        self.drop = nn.Dropout(0.5); self.fc2 = nn.Linear(hidden, n)
    def forward(self, x): return self.fc2(self.drop(self.relu(self.fc1(x))))


model = DialectNet(input_dim, HIDDEN, len(classes))
class_w = torch.ones(len(classes))
if "standard" in class_to_id and args.weight_standard != 1.0:
    class_w[class_to_id["standard"]] = args.weight_standard
    print(f">> standard class loss weight = {args.weight_standard}")
loss_fn = nn.CrossEntropyLoss(weight=class_w)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
def accuracy(lg, t): return (lg.argmax(1) == t).float().mean().item()

# 5. TRAIN (early stopping)
n = X_train_t.shape[0]
best_loss, best_state, best_epoch, since = float("inf"), None, 0, 0
for epoch in range(1, EPOCHS + 1):
    model.train(); perm = torch.randperm(n)
    for i in range(0, n, BATCH):
        idx = perm[i:i + BATCH]
        loss = loss_fn(model(X_train_t[idx]), y_train_t[idx])
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        vloss = loss_fn(model(X_val_t), y_val_t).item(); vacc = accuracy(model(X_val_t), y_val_t)
    if vloss < best_loss - 1e-4:
        best_loss, best_epoch, since = vloss, epoch, 0; best_state = copy.deepcopy(model.state_dict())
    else: since += 1
    if epoch % 10 == 0 or epoch == 1:
        with torch.no_grad(): tacc = accuracy(model(X_train_t), y_train_t)
        print(f"epoch {epoch:3d} | train acc {tacc:.3f} | val loss {vloss:.3f} acc {vacc:.3f}"
              + ("  <- best" if best_epoch == epoch else ""))
    if since >= PATIENCE: print(f"\nEarly stop at epoch {epoch}."); break

model.load_state_dict(best_state)
with torch.no_grad(): val_acc_best = accuracy(model(X_val_t), y_val_t)
print(f"Restored best model from epoch {best_epoch} (val acc {val_acc_best:.3f}).")


def report(yt, yp, title):
    print(f"\n=== {title} ===")
    print("         " + " ".join(f"{c[:4]:>5}" for c in classes))
    cm = confusion_matrix(yt, yp, labels=list(range(len(classes))))
    for i, c in enumerate(classes):
        print(f"{c[:8]:>8} " + " ".join(f"{cm[i][j]:5d}" for j in range(len(classes))))
    print("\n" + classification_report(yt, yp, labels=list(range(len(classes))),
                                        target_names=classes, zero_division=0))

model.eval()
with torch.no_grad(): val_preds = model(X_val_t).argmax(1).numpy()
report(y_val, val_preds, "IN-DOMAIN validation (Wikipedia, unseen articles)")

if args.test:
    test = read_csv(args.test)
    t_txt = [t[0] for t in test if t[1] in class_to_id]
    t_y = np.array([class_to_id[t[1]] for t in test if t[1] in class_to_id])
    if t_txt:
        Xt = torch.from_numpy(vectorizer.transform(t_txt).toarray().astype(np.float32))
        with torch.no_grad(): t_pred = model(Xt).argmax(1).numpy()
        print(f"\nCROSS-DOMAIN accuracy: {accuracy_score(t_y, t_pred):.3f} (vs {val_acc_best:.3f} in-domain)")
        report(t_y, t_pred, f"CROSS-DOMAIN test: {args.test} ({len(t_txt)} sentences)")

# SAVE: bundle weights + fitted vectorizer + metadata into one file so the
# classifier can be reloaded for inference without retraining (see predict.py).
if args.save:
    torch.save({
        "state_dict": model.state_dict(),
        "vectorizer": vectorizer,        # fitted sklearn FeatureUnion (pickled)
        "classes": classes,
        "input_dim": input_dim,
        "hidden": HIDDEN,
    }, args.save)
    print(f"\nSaved model bundle -> {args.save}  (reload with: python predict.py --model {args.save} \"your text\")")


def predict(text):
    vec = torch.from_numpy(vectorizer.transform([text]).toarray().astype(np.float32))
    with torch.no_grad(): probs = torch.softmax(model(vec), 1)[0]
    return sorted(zip(classes, probs.tolist()), key=lambda x: -x[1])


def interpret_prediction(text, top_n_features=3):
    """
    Diagnostic tool: Maps model coefficients to active tokens to determine
    exactly why a test sentence is being routed to a specific class.
    """
    feature_names = vectorizer.get_feature_names_out()
    vec_np = vectorizer.transform([text]).toarray().astype(np.float32)[0]
    active_indices = vec_np.nonzero()[0]
    
    if len(active_indices) == 0:
        return

    # Approximate neural pathway mapping matrix by collapsing layer projections (W_fc2 * W_fc1)
    with torch.no_grad():
        eff_weights = torch.mm(model.fc2.weight, model.fc1.weight).numpy()

    print("    -> Feature Breakdown (Token Impact Scores):")
    for cls_name in classes:
        cls_idx = class_to_id[cls_name]
        token_impacts = []
        for idx in active_indices:
            token = feature_names[idx]
            # Impact score = token absolute frequency metric * structural network importance weight
            impact_score = eff_weights[cls_idx, idx] * vec_np[idx]
            token_impacts.append((token, impact_score))
        
        # Sort by raw numerical impact score descending
        token_impacts = sorted(token_impacts, key=lambda x: -x[1])[:top_n_features]
        impact_strings = [f"'{t}': {s:.2f}" for t, s in token_impacts]
        print(f"        {cls_name:<11} top features: {', '.join(impact_strings)}")


print("\n=== Quick test ===")
for t in ["La Sicilia è na bella ìsula", "Napule è 'a cchiù bella città"]:
    top = predict(t)[0]
    print(f"  '{t}'  ->  {top[0]} ({top[1]:.0%})")
    interpret_prediction(t)