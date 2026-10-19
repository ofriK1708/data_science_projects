# Interview Context: `music_success_audio_features`

> Prep notes for a technical interview (Associate Software Engineer @ Genesys — productionizing AI models, Java/Python, AWS, testing, OOP).
> Everything below comes from what is actually in this repo (commit `23cdb48`). If a template section does not apply, it says so. **Don't claim features the repo lacks.** Instead, use the "If I productionized this" notes.

---

## 1. Overview

**What it does.** This is an exploratory data-science project. It asks whether a song's **Spotify audio features** (danceability, energy, loudness, valence, …) predict commercial **success** among songs from the past ~20 years. "Success" is a binary label made by thresholding Spotify's `Popularity` score. The final artifact is a Random Forest classifier, evaluated with out-of-bag (OOB) metrics, plus a permutation-importance analysis of which audio features matter most.

**Repo layout**

| Path | Purpose |
|---|---|
| `music_success_audio_features/researchDataset.ipynb` | **The whole project**: loading, cleaning, feature engineering, EDA, labeling, modeling, evaluation |
| `music_success_audio_features/data/song_hits_and_audio_features.csv` | Raw dataset (15,150 rows × 18 cols, committed to git) |
| `music_success_audio_features/models/songs_audio_model.ipynb` | Placeholder (`print("Hello World!")`). There is no separate model module. |
| `music_success_audio_features/final presentation.pptx` | Results presentation |
| `music_success_audio_features/requirements.txt` | **Empty** |
| `music_success_audio_features/README.md` | Title only |

**Tech stack** (inferred from imports; nothing is pinned)
- Python 3 in Jupyter. The notebook metadata says `2.7.6`, but that is stale: the code uses `OneHotEncoder(sparse_output=...)`, which needs **scikit-learn ≥ 1.2**, so it must be Python 3.
- pandas (uses the `observed=` arg of `groupby` and the `category` dtype)
- scikit-learn: `StandardScaler`, `OneHotEncoder`, `ColumnTransformer`, `RandomForestClassifier`, `permutation_importance`, `precision_recall_curve`, `classification_report`, `confusion_matrix`
- matplotlib + seaborn for plots

**How it runs / deploys.** You open the notebook and run the cells top to bottom; it reads the CSV from a relative path (`data/...`). There is **no deployment**: no saved model artifact, no API, no container, no CI.

---

## 2. Architecture

This is not a layered application. It is a linear notebook pipeline:

```
CSV ──► load & null check ──► clean/rename ──► feature engineering ──► filter era (00s, >=10s)
    ──► label (popularity >= 63) ──► EDA (correlation heatmap, success rate by genre)
    ──► Model A: RF on audio + genre/key/mode (OOB eval, threshold 0.52, PR curve, confusion matrix)
    ──► Model B: RF on audio only ──► permutation importance (top features)
```

**Key "components"** (notebook cells in `researchDataset.ipynb`; there are no classes)

| Cell | Responsibility |
|---|---|
| 2 | `pd.read_csv` and `isnull().sum()` (0 nulls in every column) |
| 4 | Column normalization, ms→minutes, scaled copies, `era` binning, engineered features |
| 6–8 | Filter to 2000+ and build the `is_success` label |
| 10 | Feature config lists (`audio_features`, categorical dtypes) |
| 12–13 | EDA: Pearson correlation with the target; success rate per genre |
| 15 | **Model A**: `ColumnTransformer` + `RandomForestClassifier(class_weight='balanced', oob_score=True)`, custom threshold, PR curve |
| 19 | `plot_confusion_matrix()` helper (the only function defined) |
| 21–22 | **Model B**: audio-only RF + `permutation_importance` |

**OOP angle for the interview.** The code has no classes. This is a good place to show design thinking: you'd wrap it in a `FeatureBuilder` (stateless transforms), a `SuccessLabeler` (threshold policy), and a `SuccessModel` (wraps a fitted sklearn `Pipeline`, exposes `predict_proba` and `predict(threshold)`, and handles `save()`/`load()`).

---

## 3. Data layer

**Not applicable as a database.** There is no DB, ORM, schema migration, query, index, or pagination. The "data layer" is one flat CSV loaded fully into memory.

**Schema (CSV columns):**
`Track, Artist, Year, Duration (ms), Time_Signature, Danceability, Energy, Key (0–11), Loudness (dB), Mode (0/1), Speechiness, Acousticness, Instrumentalness, Liveness, Valence, Tempo, Popularity (0–100), Genre`

**Implicit entities and relationships**
- Artist 1—N Song (in the modeled subset: 610 artists, 3,421 songs, about 5.6 songs per artist on average)
- Song N—1 Genre. However, the same song can appear under **several genres**: there are 279 duplicate `(Track, Artist)` pairs overall and **43 in the modeled subset**. That makes "Genre" closer to a playlist membership than a property of the song.
- Year range is 1899–2024. The modeled subset covers 2000–2024 (2024 has only 8 songs).

**Data facts worth knowing**
- 19 genres; Pop is the largest (3,669 rows).
- `Today` "genre" has 620 rows, median year 2020. It is a **recency bucket, not a musical genre** (see the leakage discussion in §7).
- There are 0 nulls, but nothing checks for duplicates (1 exact duplicate row, 279 track+artist duplicates).

**If I productionized this:** store features in a table keyed by a stable track ID (for example a Spotify ID) instead of name+artist; deduplicate at ingest; keep labels in a separate, versioned table with a `label_date`, because popularity changes over time.

---

## 4. Error handling & logging

**There is effectively none, and you should say so plainly if asked.**
- Validation is limited to `df.isnull().sum()` (a manual visual check).
- There is no schema validation, range checks (for example, that `danceability` is in [0,1]), or dedup.
- `OneHotEncoder(handle_unknown='ignore')` is the one piece of defensive design: an unseen genre/key at inference becomes all zeros instead of crashing.
- Logging is `print()` statements only.

**If I productionized this:** validate inputs with a Pydantic model (Python) or Bean Validation (Java) at the API boundary, add range/enum checks, use the `logging` module with structured JSON logs, return proper 4xx errors for bad input, and monitor feature distributions for drift.

---

## 5. Testing

**There are no tests**: no `tests/` directory, no pytest or unittest, no CI. Evaluation is model evaluation (OOB metrics), not software testing. Interviewers care about the difference.

**What I would add (good talking points):**
1. **Unit tests (pytest)** for the pure functions: ms→minutes conversion, era binning edges (1969 → `<=60s`, 1970 → `70s`), label threshold (62 → 0, 63 → 1), engineered features on a tiny hand-made DataFrame.
2. **Data contract tests**: required columns present, dtypes, value ranges, no duplicate track IDs.
3. **Model tests**: a "golden" check that metrics stay above a floor on a fixed holdout; a smoke test that a single-row prediction returns a probability in [0,1]; an invariance test (for example, duplicating an input row doesn't change its prediction).
4. **API tests** if served: contract tests on request/response schema and error codes.

---

## 6. Integrations

**Not applicable. There is no auth (Cognito/JWT), no external API, no OpenAI/LLM function calling, and no AWS service.** The dataset is a static CSV. Spotify features were presumably pulled beforehand, but the extraction code is not in the repo.

**If asked "how would you put this on AWS?"** a reasonable answer:
- Train offline, then `joblib.dump` the **entire sklearn `Pipeline`** (preprocessor + model) and save it to **S3**, versioned and with metadata (training data hash, threshold, metrics).
- Serve with **FastAPI in a container on ECS/Fargate** or a **SageMaker endpoint**. Auth goes through API Gateway with a Cognito/JWT authorizer.
- Logs and metrics go to CloudWatch. Record the prediction distribution and input feature stats for drift monitoring.
- Tie it into CI (GitHub Actions): tests, then train or evaluate, then a gate on the metric floor, then deploy.

---

## 7. ML details (the core of this project)

### Data source and size
- `song_hits_and_audio_features.csv`: **15,150 songs**, 18 columns, Spotify audio features + Spotify `Popularity` + genre label.
- Modeled subset: **3,421 songs** from eras `00s` and `>=10s` (years 2000–2024).

### Preprocessing and feature engineering (cell 4)
```python
df = df.rename(columns={"Track": "name"})
df.columns = df.columns.str.lower()
df["duration"] = (df["duration"] / 60000).round(2)          # ms -> minutes

scaler = StandardScaler()
df['loudness_scaled'] = scaler.fit_transform(df[['loudness']])
df['energy_scaled']   = scaler.fit_transform(df[['energy']])
# ... acousticness, danceability, valence likewise
df["era"] = pd.cut(df["year"], bins=[1889, 1969, 1979, 1989, 1999, 2009, 2029],
                   labels=["<=60s", "70s", "80s", "90s", "00s", ">=10s"])

df['intensity']          = df[['energy_scaled', 'loudness_scaled']].mean(axis=1)
df['feel_good_factor']   = df[['danceability_scaled', 'valence_scaled']].mean(axis=1)
df['production_balance'] = df['energy_scaled'] - df['acousticness_scaled']
```
The reasoning behind the engineered features: energy and loudness are highly correlated, so averaging them gives one "intensity" axis. Danceability plus valence gives a "feel-good" axis. Energy minus acousticness captures the trade-off between electronic and acoustic production.

> Note: the scaler is **fit on the full 15k dataset before filtering and before any split**. That doesn't matter for a Random Forest (trees are scale-invariant), but it is a train/serve-consistency smell.

### Label definition (cells 6–8)
```python
ABSOLUTE_SUCCESS_THRESHOLD = 63      # ~70th percentile of popularity in 2000+ songs
songs_past_20_years['is_success'] = (
    songs_past_20_years['popularity'] >= ABSOLUTE_SUCCESS_THRESHOLD).astype(int)
# -> 0: 2368, 1: 1053  (~31% positive, imbalanced)
```

### Features
- **Model A (final):** 9 numeric audio features (`danceability, valence, speechiness, acousticness, instrumentalness, liveness, tempo, energy, loudness`), plus one-hot encoded `genre`, `key`, `mode`.
- **Model B:** the 9 audio features only, used for feature importance.
- Engineered features (`intensity`, `feel_good_factor`, `production_balance`) were **tried and dropped**. OOB accuracy was 0.764 with them vs **0.780** with the raw features (documented in markdown cells 17–18).

### Model (cell 15)
```python
preprocessor = ColumnTransformer(transformers=[
    ('audio', StandardScaler(), audio_features),
    ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
     categorical_features)], remainder='passthrough')
X_transformed = preprocessor.fit_transform(X)

model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1,
                               oob_score=True, class_weight='balanced')
model.fit(X_transformed, y)

threshold = 0.52
oob_probabilities = model.oob_decision_function_[:, 1]
y_oob_pred = (oob_probabilities >= threshold).astype(int)
print(classification_report(y, y_oob_pred, target_names=['Not Success', 'Success']))
```

### Train/test split method
**There is no explicit split, neither random nor time-based.** Evaluation uses **Random Forest out-of-bag (OOB) estimates**: each tree trains on a bootstrap sample, and each row is scored only by the trees that did not see it (about 37% of trees). That is roughly equivalent to a **random** holdout. It is **not time-based**, even though the data covers 2000–2024 and the label (popularity) depends heavily on recency.

### Metrics (OOB, Model A, threshold 0.52)
| | precision | recall | f1 | support |
|---|---|---|---|---|
| Not Success | 0.80 | 0.90 | 0.85 | 2368 |
| **Success** | **0.69** | **0.50** | **0.58** | 1053 |
| accuracy | | | **0.78** | 3421 |

- Baseline for comparison: always predicting "Not Success" gives **0.69 accuracy** (2368/3421). The real lift is in success-class precision and recall, not accuracy.
- **Model B (audio only):** OOB accuracy 0.702, success precision 0.532. Genre adds a lot.
- **Permutation importance (Model B), top 3:** energy 0.148, danceability 0.119, instrumentalness 0.116.
- **Genre success rate (EDA):** Today 0.69, Alt. Rock 0.52, R&B 0.51, Rap 0.46 … Jazz/Blues/Folk/Disco/World 0.00.

### What matters for serving in production
1. **Persist one fitted `Pipeline(preprocessor, model)`**, not separate objects. Today the preprocessor and model are separate variables and nothing is saved, so the run can't be reproduced or served.
2. **Threshold is a business parameter** (0.52, picked off the PR curve). Treat it as configuration, version it with the model, and choose it from an explicit precision/recall target.
3. **Label drift**: Spotify Popularity is recency-weighted and changes daily. A label snapshot ages, so retrain on a schedule and record the label date.
4. **Unseen categories**: `handle_unknown='ignore'` keeps inference from crashing on a new genre, but that prediction degrades silently. Log and alert on it.
5. **Input contract**: the model needs `genre`, which a brand-new song may not have. Audio features come from an upstream API, so define the schema and fallbacks.
6. **Latency/size**: 150 trees is fine for a CPU microservice (single-row inference takes milliseconds); a batch scoring job also works.

---

## 8. Strongest decisions, trade-offs, and weaknesses

### Strong decisions (lead with these)
- **Handled class imbalance deliberately**: `class_weight='balanced'` plus a **tuned decision threshold** taken from a precision-recall curve, instead of relying on accuracy at 0.5.
- **Used a `ColumnTransformer`** to keep numeric and categorical preprocessing declarative, and `handle_unknown='ignore'` to avoid inference-time crashes.
- **Permutation importance** instead of only impurity-based `feature_importances_`, which is biased toward high-cardinality and continuous features.
- **Ran an ablation**: engineered features vs. raw, with and without genre. The engineered features lost (0.764 vs 0.780) and were dropped instead of kept for show.
- **Absolute, percentile-informed label threshold** (63 ≈ P70), with the reasoning documented.
- `random_state=42` set on the model, so its training is reproducible.

### Weaknesses / tech debt an interviewer will likely probe
| Issue | Why it matters | Fix |
|---|---|---|
| **Target leakage via `Today` genre** | "Today" is a recency bucket (median year 2020), and Spotify popularity favors recent songs. The genre with the top success rate (0.69) is partly encoding the label. | Drop or rename the feature, or split by time; compare metrics with and without it |
| **No time-based split** | Random/OOB evaluation mixes future songs into training, which overstates real "will a new song succeed?" performance | Train on ≤2018, test on 2019+ |
| **Artist leakage** | 97% of songs share an artist with another song, so the model can learn artist/genre signatures | `GroupKFold` by artist |
| **Duplicate songs across genres** (43 in subset) | The same song can be in-bag and OOB at once, which inflates OOB metrics | Dedupe on (track, artist), or merge genres into multi-hot |
| **Preprocessor fit on all data** | Classic leakage pattern (harmless for RF here, but it's a habit to avoid) | Put everything in a `Pipeline` and cross-validate it |
| **Permutation importance on training data** | Importances computed on data the model has seen are biased | Compute them on a holdout set |
| **Threshold picked on the same OOB data used to report metrics** | Optimistic estimate | Tune on validation data, report on test data |
| **Markdown out of sync with code** | Cell 17 lists `duration`, `mode`, `key` as audio features, but `audio_features` doesn't include `duration`. Model B omits key/mode. The comment says "0.53 threshold" but the code uses 0.52. | Generate reports from code |
| **No reproducibility infrastructure** | Empty `requirements.txt`, stale kernel metadata (2.7.6), no saved model, CSV in git | Pin deps, save artifacts, use DVC/S3 for data |
| **No tests, no modules, no logging** | Not production-ready | See §4–5 |
| **No hyperparameter tuning or baseline model** | We don't know whether RF is the right choice | Compare against logistic regression; run `RandomizedSearchCV` |

---

## 9. Ten likely deep-dive questions (with answer pointers)

1. **"Walk me through how you defined 'success'. Why 63?"** It's the ~70th percentile of popularity in 2000+ songs, which gives about 31% positives. Discuss the trade-off between absolute and relative thresholds, and that popularity is a moving, recency-biased snapshot.
2. **"How did you evaluate the model? Why OOB instead of a train/test split?"** OOB acts as a free, roughly unbiased holdout for bagged trees. Then admit the gaps: it's random, not time- or artist-grouped, and the threshold was tuned on the same estimates.
3. **"Accuracy is 0.78. Is that good?"** The majority-class baseline is 0.69. The meaningful numbers are success precision 0.69 and recall 0.50, and which of those the business cares about.
4. **"How did you pick the 0.52 threshold, and how would you choose it in production?"** From the PR curve. In production it should come from a target (for example, precision ≥ 0.7), be tuned on validation data, and ship as versioned config.
5. **"Is there any leakage in this setup?"** The `Today` genre, artist overlap, duplicate songs, and no time split. This is the most important question to answer well.
6. **"Why a Random Forest? What else would you try?"** It handles nonlinear effects and mixed features with little tuning, and gives OOB estimates for free. Alternatives: logistic regression (interpretable baseline), gradient boosting (LightGBM/XGBoost), calibrated probabilities.
7. **"Your engineered features performed worse. Why might that be?"** Trees can already learn interactions; averaging loses information; scaling doesn't help trees. Mention that the features came from the correlation analysis.
8. **"Permutation vs impurity importance: what's the difference?"** Impurity-based importance is biased toward high-cardinality features and computed on training data. Permutation importance measures the metric drop when a feature is shuffled; it should run on held-out data, and correlated features split credit between them.
9. **"How would you take this notebook to a production service at Genesys?"** Refactor into a package (OOP: `FeatureBuilder`/`SuccessModel`), use a single sklearn `Pipeline`, `joblib` the artifact to S3, serve with FastAPI on ECS/SageMaker, add pytest + CI, a metric gate, CloudWatch logging, and drift monitoring. A Java service could call it over REST, or the model could be exported to ONNX.
10. **"How would you test an ML system?"** Unit-test the transforms and label logic; data contract tests; model regression tests with a metric floor; invariance and directional tests; API contract tests. Make clear that offline evaluation is not the same as software tests.

**Bonus follow-ups to prepare:** handling a new genre at inference (`handle_unknown='ignore'` and its silent-degradation risk); retraining cadence given popularity drift; how you'd detect data drift (comparing feature distributions, e.g. PSI or KS tests).
