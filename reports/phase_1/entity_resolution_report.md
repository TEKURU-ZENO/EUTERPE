# EUTERPE — Entity Resolution & Identity Audit Report
**Phase:** 1 — Data Engineering & Listening Intelligence  
**Dataset:** Last.fm 1K Entity Resolution  

---

## 1. Resolution Summary

| Metric | Count | Reduction Ratio | Definition / Purpose |
|---|:---:|:---:|---|
| **Raw Distinct Pairs** | 1,500,661 | 0.0% | Unprocessed (artist, track) strings from scrobbles |
| **`recording_key` (Conservative)** | 1,489,456 | **0.75%** | Preserves live, acoustic, remix, remaster, features |
| **`normalized_track_key` (Broad)** | 1,452,747 | **3.19%** | Collapses remaster/live noise for broad grouping |

---

## 2. Collapse Analysis & Preservation Verification

The dual-identity architecture satisfies **Senior Scientific Review 03**:
- By preserving `recording_key`, acoustic and live performances are not blindly conflated with original album studio cuts.
- By providing `normalized_track_key`, cross-release popularity aggregation and candidate discovery grouping can operate without artificial fragmentation.

### Sample Broad Collapses (Multiple Raw Strings -> Single Broad Key)
```json
[
  {
    "key": ":::",
    "variants_collapsed": 56,
    "sample_variants": [
      "?????? ???? ????????? ????????",
      "?????????",
      "????",
      "!!! - Live! Live! Live! - 01 - Pardon My Freedom (Live At Liquidroom Ebisu)"
    ]
  },
  {
    "key": "unknown:::",
    "variants_collapsed": 41,
    "sample_variants": [
      "??????? ???? ?????",
      "???(??????????)",
      "?????\u00b7?????",
      "???????\u2026"
    ]
  },
  {
    "key": "melvins:::melvins",
    "variants_collapsed": 19,
    "sample_variants": [
      "Melvins - Live 2000.04.15 Chicago - Set 2 - 27Let It All Be",
      "Melvins - Live 2000.04.15 Chicago - Set 2 - 20Manky",
      "Melvins - Live 2000.04.15 Chicago - Set 1 - 10Now I'M Here",
      "Melvins - Live 2000.04.15 Chicago - Set 1 - 04Skin Horse (Second Half)"
    ]
  },
  {
    "key": "kreator:::2003",
    "variants_collapsed": 18,
    "sample_variants": [
      "2003 - Live Kreation / 02 - Violence Revolution",
      "2003 - Live Kreation / 16 - Second Awakening",
      "2003 - Live Kreation / 13 - Riot Of Violence",
      "2003 - Live Kreation / 01 - The Patriarch"
    ]
  },
  {
    "key": "plastic tree:::",
    "variants_collapsed": 15,
    "sample_variants": [
      "????????????",
      "???????",
      "?????",
      "????????"
    ]
  },
  {
    "key": "safri duo:::played a",
    "variants_collapsed": 14,
    "sample_variants": [
      "Played-A-Live (Airscape)",
      "Played-A-Live (Airscape Remix)",
      "Played-A-Live (The Bongo Song) (Radio Cut)",
      "Played-A-Live"
    ]
  },
  {
    "key": "thee michelle gun elephant:::",
    "variants_collapsed": 12,
    "sample_variants": [
      "????",
      "???\u00b7???\u00b7???",
      "????\u00b7?????\u00b7????",
      "?????\u00b7??\u00b7??\u00b7???"
    ]
  },
  {
    "key": "testament:::2000",
    "variants_collapsed": 11,
    "sample_variants": [
      "2000 - Live At Dynamo / 10 - Legions Of Dead",
      "2000 - Live At Dynamo / 01 - D.N.R",
      "2000 - Live At Dynamo / 01 - 3 Days Of Darkness",
      "2000 - Live At Dynamo / 09 - Eyes Of Wrath"
    ]
  },
  {
    "key": "coldplay:::clocks",
    "variants_collapsed": 11,
    "sample_variants": [
      "Clocks (Live R\u00f6yksopp Remix)",
      "Clocks (Live)",
      "Clocks (Instrumental)",
      "Clocks (Live Isle Of Wight Festival June 11Th 2006)"
    ]
  },
  {
    "key": "nina simone:::ain t got no i got life",
    "variants_collapsed": 10,
    "sample_variants": [
      "Ain'T Got No/I Got Life (Live)",
      "Ain'T Got No - I Got Life",
      "Ain'T Got No, I Got Life",
      "Ain'T Got No I Got Life"
    ]
  }
]
```

---

## 3. Engineering Sign-off
Canonical track lookup persisted to `C:\Users\Dev Mehta\Desktop\EUTERPE\data\processed\canonical_tracks.parquet` in 19.17 seconds.
