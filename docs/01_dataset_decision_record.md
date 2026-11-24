# EUTERPE — Dataset Decision Record & Data Ingestion Specification
**Version:** 1.0.0  
**Phase:** 0 — Scientific Specification and Research Foundation  
**Primary Dataset:** Last.fm Dataset - 1K Users (`lastfm-dataset-1K`)  

---

## 1. Dataset Selection & Comparative Evaluation

To establish a reproducible, scientifically rigorous foundation for EUTERPE, candidate music recommendation corpora were evaluated across four dimensions: **temporal fidelity**, **implicit feedback density**, **metadata quality**, and **computational tractability**.

| Dataset | Candidate Size | Time Range | Timestamp Granularity | Track Identity | Pros / Cons | Decision |
|---|---|---|---|---|---|---|
| **Last.fm 1K** (`lastfm-dataset-1K`) | 19.15M scrobbles, 992 users | 2005–2009 | Exact second (ISO 8601) | Artist + Track name (+ partial MBID) | **Pros:** Complete longitudinal history per user; realistic scrobble sequences; accessible archive.<br>**Cons:** Track MBIDs sparse. | **SELECTED as Primary Benchmark** |
| **Million Song Dataset (Taste Profile)** | 48M interactions, 1.01M users | Static aggregate | No exact event timestamp (play counts only) | Echo Nest Song ID | **Pros:** Large volume.<br>**Cons:** Lacks temporal scrobble order; impossible to do true sequential/temporal split. | Rejected for Phase 0–6 |
| **LFM-2b** | 2 billion scrobbles | 2005–2020 | Microsecond timestamps | Internal IDs + Spotify/MBID mapping | **Pros:** Massive real-world scale.<br>**Cons:** >100 GB compressed; unfeasible for fast iterative model benchmarking. | Reserved for Phase 8–9 |

### Decision Summary
`lastfm-dataset-1K` is selected as the primary benchmark. Its complete longitudinal history per user enables strict **temporal splitting** without data leakage, which is essential to valid offline evaluation in recommender systems.

---

## 2. Dataset Schema & Empirical Properties

The dataset is distributed by the Music Technology Group (MTG) at Universitat Pompeu Fabra (UPF) in two tab-separated (`.tsv`) files:

### File 1: Listening Events (`userid-timestamp-artid-artname-traid-traname.tsv`)
* **Size:** 2.53 GB uncompressed (19,150,868 rows)
* **Schema (6 fields, tab-delimited):**
  1. `userid` (`str`): User identifier (e.g., `user_000639`).
  2. `timestamp` (`str`): ISO 8601 UTC timestamp (e.g., `2009-04-08T01:57:47Z`).
  3. `musicbrainz-artist-id` (`str`, optional): UUID from MusicBrainz (e.g., `15676fc4-ba0b-4871-ac8d-ef058895b075`).
  4. `artist-name` (`str`): Raw artist name (UTF-8, multilingual).
  5. `musicbrainz-track-id` (`str`, optional): UUID from MusicBrainz.
  6. `track-name` (`str`): Raw track title (UTF-8).

### File 2: User Profiles (`userid-profile.tsv`)
* **Size:** 37.8 KB (992 rows)
* **Schema (5 fields, tab-delimited):**
  1. `userid` (`str`): Unique user identifier.
  2. `gender` (`str`): Categorical (`m`, `f`, or empty).
  3. `age` (`int`, optional): Self-reported age.
  4. `country` (`str`, optional): Country of residence.
  5. `signup` (`str`, optional): User registration date (e.g., `Apr 27, 2005`).

---

## 3. Data Quality Insights & Mitigation Protocols

Empirical analysis of the raw stream revealed critical data engineering challenges that must be handled deterministically:

### 3.1 MusicBrainz Track ID Sparsity
* **Finding:** While `musicbrainz-artist-id` is present for ~60.8% of unique artists (107,528 of 176,948), `musicbrainz-track-id` is missing in **>80%** of interaction rows.
* **Mitigation (Canonical Identity Resolution):** Tracks cannot be keyed by MBID alone. A canonical composite key is defined:
  $$\text{TrackKey} = \text{Norm}(\text{artist\_name}) \mathbin{\Vert} \text{":::"} \mathbin{\Vert} \text{Norm}(\text{track\_name})$$
  where $\text{Norm}(\cdot)$ performs:
  1. Lowercase transformation and Unicode NFKD normalization (removing accents/diacritics).
  2. Stripping common acoustic qualifiers: `[feat. ...]`, `(feat. ...)`, `(remastered ...)`, `(live ...)`, `(album version)`.
  3. Punctuation removal and internal whitespace collapse.
  4. Preservation of pristine canonical display strings (`display_artist`, `display_title`) alongside the normalized key.

### 3.2 Reverse-Chronological Ordering in Raw Dumps
* **Finding:** Scrobbles for each user are emitted in reverse-chronological order (from 2009 backwards to 2005) because they were harvested via Last.fm's `user.getRecentTracks` API endpoint.
* **Mitigation:** Raw ingestion must sort events strictly by `timestamp ASC` per user prior to interaction aggregation or splitting.

### 3.3 Repeat-Scrobble Deduplication & Spam Filtering
* **Finding:** Erroneous media players occasionally log duplicate scrobbles for the same track within 10–30 seconds.
* **Mitigation:** Interactions where $\Delta t = t_{k} - t_{k-1} < 30\text{ seconds}$ for identical track keys by the same user are dropped as duplicate client telemetry.

### 3.4 Interaction Thresholds & Core Filtering
To prevent extreme sparsity artifacts while preserving the natural long tail:
* **User filter:** Active users with $\ge 50$ total scrobbles across their history.
* **Track filter:** Items with $\ge 5$ scrobbles across all users (5-core item filter).
* **Confidence transformation:** Play counts $r_{ui} = \sum_{k} \mathbb{I}(u, i)$ transformed via log-scaling:
  $$c_{ui} = 1 + \alpha \cdot \log(1 + r_{ui})$$
  with $\alpha = 40.0$ to prevent high-frequency loop listening from dominating preference vectors.

---

## 4. Licensing, Ethics & Attribution

* **License:** Distributed under non-commercial research permission granted by Last.fm.
* **Attribution Requirement:**
  > Celma, O. (2010). *Music Recommendation and Discovery in the Long Tail*. Springer Berlin Heidelberg.
* **Data Privacy:** Usernames are anonymized into opaque identifiers (`user_000001` through `user_001000`). No PII or raw IP addresses are contained in the dataset.
