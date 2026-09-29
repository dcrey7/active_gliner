---
title: Dataset sources - pinned, downloaded and checked for the 7 task datasets
date: 2026-09-25 12:30 CEST
author: Claude (subagent)
type: research
status: frozen
---

# Dataset sources

Question: for each dataset in the design (section 2 and section 13 of
`docs/research/2026-09-25-1113-full-merge-design.md`), where is a pinned source
that loads with `datasets` 5.0.1 (no loading scripts), and what does the data
look like?

All raw files are in `~/.cache/active_gliner/raw/<dataset>/`, not in the repo.
All numbers below come from scripts I ran on those files on 25 Sep 2026.
The scripts are in `/tmp/ag_*.py` (not kept in the repo).
Nothing was uploaded.

## Summary

| # | Dataset | Ready? | Pinned source | Train / dev / test (sentences) |
|---|---|---|---|---|
| 1 | CleanCoNLL | Built with the official script | `flairNLP/CleanCoNLL` commit `bedc569f` + CoNLL-03 zip from `data.deepai.org` | 13,957 / 3,233 / 3,427 |
| 2 | BC5CDR | Yes, with PMIDs rebuilt | `tner/bc5cdr` + original `CDR_Data.zip` (in `bigbio/bc5cdr`) | 5,228 / 5,330 / 5,865 |
| 3 | MIT Movie fixed | Yes | `rungalileo/mit_movies_fixed_connll_format` `bf6c430a` | 9,775 / none / 2,443 |
| 4 | CrossRE | Yes | `mainlp/CrossRE` commit `a58885fa` (HF parquet is identical) | 668 / 2,151 / 2,446 |
| 5 | Hallmarks of Cancer | Yes | `bigbio/hallmarks_of_cancer` parquet `b78c5a2c` | 12,119 / 1,798 / 3,547 |
| 6 | MASSIVE 1.1 en-US | Yes | `amazon-massive-dataset-1.1.tar.gz` sha256 `4cba5faa...` | 11,514 / 2,033 / 2,974 |
| 7 | MASSIVE 1.1 fr-FR | Yes | same tarball | 11,514 / 2,033 / 2,974 |

Main findings:

+ MIT Movie in the repo - `data/mit-movie/*.json` holds the original MIT labels, not the Galileo fix. Its `dev.json` is an exact copy of `test.json`.
+ CrossRE - 1,317 of 5,265 sentences have no relation. Not every sentence carries a relation.
+ BC5CDR - the TNER copy has no ids, but every sentence maps back to one of 500 PMIDs per split. Mention counts match the original corpus exactly.
+ CleanCoNLL - the build needs the unofficial `data.deepai.org` mirror of CoNLL-03. The Reuters text licence is the open point.
+ Script-only repos - `bigbio/bc5cdr` has no parquet conversion. `DFKI-SLT/cross_re` and `bigbio/hallmarks_of_cancer` have one.

## 1. CleanCoNLL (NER)

How the build works (source: `README.md` and `create_cleanconll_from_conll03.sh` at commit `bedc569f`):

1. The repo ships annotation files with masked tokens (`[TOKEN]`).
2. The script downloads `https://data.deepai.org/conll2003.zip` with `wget`.
3. It applies three small `patch` files to the CoNLL-03 token column. The patches move the text to the Reiss et al. (2020) corrected token base.
4. It pastes the patched tokens next to the annotation columns.

No registration is needed: the deepai URL returned HTTP 200 without login.
I pre-placed the zip, so the script skipped `wget` and ran the rest unchanged.
All three patches applied cleanly.
Line counts of patched tokens and annotations match (219,505 / 55,047 / 50,392).

| Field | Value |
|---|---|
| Annotation source | `https://github.com/flairNLP/CleanCoNLL`, commit `bedc569f0983ddd39c2c729fb5df64be710bcab4` (2024-07-02), archive sha256 `c07ada43e79b99d461119ce4fef84899828818bb3034a3c0fd352c6e3a650df4` |
| Text source | `https://data.deepai.org/conll2003.zip`, sha256 `96a104d174ddae7558bab603f19382c5fe02ff1da5c077a7f3ce2ced1578a2c3`; metadata says `name: ner-conll2003-eng` |
| Built files | `~/.cache/active_gliner/raw/cleanconll/CleanCoNLL-bedc569f.../data/cleanconll/cleanconll.{train,dev,test}` |
| Built sha256 | train `be721578a9d39b5e0f2c37a0db5d5aaf61eb8e6a36a6bd2cb5e57316c5651e09`, dev `d2a02b05a371a277bf0bc0be010f6545317dd8393df3569933868691f9f2c7f0`, test `a3d3cb8def17bd85b3fed4a8d3b3d7293eb17171190b3642bef3762a88d908fc` |
| Licence | CleanCoNLL repo has no licence file (GitHub licence API returned 404). The README says tokens are masked "for licence reasons". CoNLL-03 text is Reuters RCV1 text; its official route is the NIST Reuters corpus agreement (unconfirmed for this study). The deepai zip is an unofficial mirror. |
| Splits | train, dev, test. Dev exists. |
| Documents | 946 / 216 / 231 (`-DOCSTART-` lines), same as original CoNLL-03 |
| Sentences | 13,957 / 3,233 / 3,427 (original CoNLL-03: 14,041 / 3,250 / 3,453; the Reiss fixes merge some sentences) |
| Tokens | 203,657 / 51,383 / 46,504 |
| Labels (column 5, final CleanCoNLL) | train ORG 6,960, PER 6,527, LOC 6,455, MISC 3,624 (23,566); dev PER 1,829, ORG 1,623, LOC 1,531, MISC 983 (5,966); test ORG 1,909, PER 1,591, LOC 1,413, MISC 812 (5,725) |
| Sentences with no entity | 2,898 / 639 / 667 |
| Duplicate sentence texts | 1,338 / 179 / 266 (mostly sports score lines, unconfirmed by hand) |
| Tagging | BIO, 5 tab columns: token, POS, Wikipedia link, NER before phase 3 (CleanCoNLL*), NER final |

Record format (first lines of `cleanconll.train`):

```
-DOCSTART-	-X-	O	O	O

EU	NNP	B-European_Union	B-ORG	B-ORG
rejects	VBZ	O	O	O
German	JJ	B-Germany	B-LOC	B-MISC
```

Problems:

+ Use column 5 - column 4 is the pre-phase-3 label; column 5 is the released CleanCoNLL label (README).
+ Text source - the build depends on a third-party mirror. If it disappears, the user needs the official CoNLL-03 files.
+ No empty sentences and no malformed rows (every row has 5 columns).

## 2. BC5CDR (NER, Chemical + Disease)

| Field | Value |
|---|---|
| Sentence source | HF `tner/bc5cdr` main `f68cdc7db924369241e7868656f583072acd4e90`. It has a loading script (`bc5cdr.py`) but also plain JSON-lines files in `dataset/`. Parquet conversion `refs/convert/parquet` = `564118257f090b2592b7b1607ec60f40948acf1c`. |
| File sha256 (`dataset/*.json`) | train `62fe247960ff1b4270cb71500fb99748aab371c2dce0ef5b048fc44cb0f82ec8`, valid `d74d511b8708c645c51e98bfd3f191d54ec83a601b3c5dde84a25ad4363f7d8e`, test `20a04588b1a67c65203df698709a4384e15dac1597aa1cfb52365e7d0d6708b5`, label.json `d1d6998c78bc510b526538212c3b8eee97f1dd45ded428fc1965139c368ed59f` |
| Parquet vs JSON | identical tokens and tags in all 3 splits (checked row by row) |
| Document source | `CDR_Data.zip` inside HF `bigbio/bc5cdr` main `6ba1463320d003e8232dfa0c3ee9aaa2559998ec`, sha256 `0a359a7f038d283a7b05b084fa73de014e7410e1f5d7034bf3fd01f016fc2444`. It holds the original `CDR.Corpus.v010516` PubTator and BioC files. `bigbio/bc5cdr` itself is script-only: its parquet ref has no parquet files. |
| Licence | Original corpus README: NCBI "PUBLIC DOMAIN NOTICE", a "United States Government Work", no restriction on use; cite Wei et al. 2015 and Li et al. 2015. BigBio card: `PUBLIC_DOMAIN_MARK_1p0`. TNER card: `other`. PubMed abstract text copyright is a separate question (unconfirmed). |
| Splits | train, validation (TNER name `valid`), test. Dev exists. |
| Documents | 500 / 500 / 500 PMIDs (PubTator files) |
| Sentences | 5,228 / 5,330 / 5,865 |
| Labels | train Chemical 5,203, Disease 4,182; valid Chemical 5,347, Disease 4,244; test Chemical 5,385, Disease 4,424 |
| Match to original | Mention counts per type equal the PubTator gold counts in every split. |
| Tags | `{"O": 0, "B-Chemical": 1, "B-Disease": 2, "I-Disease": 3, "I-Chemical": 4}` |
| Sentences with no entity | 1,313 / 1,441 / 1,725 |
| Duplicate sentence texts | 87 / 128 / 152 |

Record format (`dataset/train.json`, line 1; PMID 227508):

```
{"tags": [1, 0, 0, 0, 0, 0, 1, 0], "tokens": ["Naloxone", "reverses", "the", "antihypertensive", "effect", "of", "clonidine", "."]}
```

PMID mapping. TNER rows are in document order.
I walked the rows in order and found each one in the PubTator text (title + abstract, spaces removed).
Result: 5,224 / 5,328 / 5,865 rows map directly; all 500 PMIDs per split are covered.
The 6 other rows (4 train, 2 valid) have text that is not in the source.
They lost words at the sentence start, for example `of the patients experienced sedation ...`.
Their PMID can be taken from the neighbour rows (unconfirmed for rows at a document edge).

Problems:

+ Bad sentence splits - the TNER splitter breaks at decimal points: `..., 0 .` then `2 to 2 mg / kg .` (PMID 227508). 125 / 128 / 167 rows have 3 tokens or fewer (for example `v .`, `d .`).
+ Lost words - 6 rows miss leading words (see above).
+ Tokenisation - some tokens keep punctuation (`ifosfamide,`, `%).`).
+ Fix option - rebuild sentences from PubTator with a better splitter. That changes the TNER sentence set; say so if chosen.

## 3. MIT Movie fixed (NER)

| Field | Value |
|---|---|
| Source | HF `rungalileo/mit_movies_fixed_connll_format` main `bf6c430a8673a2305638576a61f99efdd4f7b2a1` (plain TSV, no script; parquet ref points to the same commit) |
| File sha256 | train `c1f83b8d2f5c0aa1884c7414b0143930adef87bc7543063dfe239ab230a740b2`, test `aae58cc44fe1acae4e348226130d77205faa6225ca288fa3816d579fc9403a2b` |
| Licence | Card says `unknown`. Original MIT data page gives no licence (unconfirmed). |
| Splits | train, test. No dev. |
| Sentences | 9,775 / 2,443 |
| Tagging | BIOES (`B`, `I`, `E`, `S`, `O`) |
| Labels train | GENRE 4,378, ACTOR 3,228, YEAR 2,862, TITLE 2,386, RATING 2,009, PLOT 1,924, RATINGS_AVERAGE 1,875, DIRECTOR 1,726, CHARACTER 385, SONG 244, REVIEW 218, TRAILER 113 (21,348) |
| Labels test | GENRE 1,125, ACTOR 825, YEAR 723, TITLE 566, RATING 502, PLOT 483, RATINGS_AVERAGE 461, DIRECTOR 447, CHARACTER 91, SONG 52, REVIEW 48, TRAILER 30 (5,353) |
| Sentences with no entity | 55 / 13 |
| Duplicate texts | 43 / 1 |

Record format (`MIT_movies_fixed_train.tsv`, token TAB tag, blank line between sentences):

```
what	O
movies	O
star	O
bruce	B-ACTOR
willis	E-ACTOR
```

Problems:

+ Two malformed rows in train - lines 22713 (`and O`) and 38906 (`movie   O`) use spaces, not a tab. Split on any whitespace.
+ 3 BIOES order errors in train (an `I` or `E` without a matching `B`), 0 in test.
+ Galileo changed some text too: 21 repo train sentences are not found in the Galileo train file.

How the repo's `data/mit-movie/*.json` relates (I also fetched the original MIT files `engtrain.bio` sha256 `68d90d7b29bc7dbf54efce6dfddbcdcb685293a00eaf76cd48c3acb5d3b68fe8` and `engtest.bio` sha256 `2ba7c6da2e5c1d897ebbcbcefad9e0f03eed41b183106e7f26b7f24b704282fd` from `https://groups.csail.mit.edu/sls/downloads/movie/`):

| Repo file | Rows | Same entities as original MIT | Same entities as Galileo |
|---|---|---|---|
| `train.json` | 9,774 | 9,765 | 9,464 |
| `test.json` | 2,442 | 2,442 | 2,334 |
| `dev.json` | 2,442 | exact copy of `test.json` | - |

+ The repo JSON is the original MIT Movie (eng) set, not the Galileo fix. Entity counts match the original closely (for example actor 3,220 in both).
+ The repo drops 1 train row and 1 test row versus the original (9,775 and 2,443).
+ The repo `dev.json` equals `test.json`. Any thesis dev decision was made on test (unconfirmed how the thesis used it).
+ Galileo changes the entity set in 280 of 9,732 matched train sentences and 108 of 2,442 test sentences.

## 4. CrossRE (relations, 6 domains)

| Field | Value |
|---|---|
| Source | `https://github.com/mainlp/CrossRE` commit `a58885fa760559d2dc9e176e730fe731f020d6f3` (2024-08-20), archive sha256 `d04d8b1ddde0b0a8a1050bf52f594c4994f98d0ae088018c6e49788c6bc963e9`, folder `crossre_data/<domain>-<split>.json` |
| HF copy | `DFKI-SLT/cross_re` main `eb583481a1fba449b36686456c60afa80cf8c7c3` is script-only and points to the unpinned `main` branch on GitHub. Parquet ref `ff705db977307983178d66c63073921ce642aab1` has 18 files. Tokens and relation counts equal the pinned upstream files in all 18 splits. |
| Licence | Upstream `LICENSE` is GNU GPL v3. HF card licence is empty. |
| Splits | train, dev (HF: `validation`), test. Dev exists. |
| Offsets | token indices, end inclusive (`[0, 0, "organisation"]` is the single token "EU"). All spans are in range. Every relation argument is an entity span. |

Per-domain sizes (sentences, relations, sentences with no relation):

| Domain | Train | Dev | Test | Entity types |
|---|---|---|---|---|
| ai | 100 (363 rel, 17 none) | 350 (1,079, 72) | 431 (1,164, 105) | 16 |
| literature | 100 (400, 11) | 400 (1,570, 29) | 416 (1,623, 47) | 19 |
| music | 100 (513, 18) | 350 (1,981, 62) | 399 (2,415, 55) | 14 |
| news | 164 (181, 98) | 350 (308, 212) | 400 (396, 234) | 5 |
| politics | 101 (536, 11) | 350 (1,874, 65) | 400 (2,124, 63) | 15 |
| science | 103 (401, 25) | 351 (1,387, 77) | 400 (1,446, 116) | 18 |
| **Total** | **668 (2,394, 180)** | **2,151 (8,199, 517)** | **2,446 (9,168, 620)** | 39 overall |

+ Train + dev = 2,819 sentences, test = 2,446. This matches the design's "about 2.8k" and "about 2.4k".
+ Not every sentence has a relation: 180 train, 517 dev, 620 test sentences have none (25% overall). News is mostly relation-free (544 of 914).
+ 17 relation types (all splits): role 4,300, physical 2,780, general-affiliation 2,364, part-of 1,747, artifact 1,408, named 1,300, temporal 1,047, related-to 902, win-defeat 850, origin 748, type-of 718, usage 376, opposite 362, topic 313, compare 215, social 204, cause-effect 127.
+ 39 entity types (mentions, all splits): misc 2,436, organisation 2,349, location 2,001, person 1,864, politicalparty 1,725, country 1,338, writer 1,245, award 1,030, musicalartist 1,009, band 882, book 836, musicgenre 835, politician 804, election 783, album 649, astronomicalobject 630, scientist 621, algorithm 437, product 429, task 424, field 421, song 399, university 394, literarygenre 391, event 381, metrics 364, researcher 354, chemicalcompound 348, academicjournal 260, poem 246, protein 211, conference 193, magazine 146, programlang 123, enzyme 112, discipline 90, chemicalelement 87, musicalinstrument 57, theory 21.
+ Multi-label pairs: a directed pair with 2 or more types occurs 1,110 times (sum over splits; politics dev 255, politics test 293).
+ Duplicate sentence texts: news 5 / 16 / 24, music 1 / 5 / 6, others 0 to 6.

Record format (`news-train.json`, line 1; relation = head start, head end, tail start, tail end, type, explanation, uncertain flag, syntax-ambiguity flag):

```
{"doc_key": "news-train-1", "sentence": ["EU", "rejects", "German", "call", "to", "boycott", "British", "lamb", "."], "ner": [[0, 0, "organisation"], [2, 3, "misc"], [6, 7, "misc"]], "relations": [[0, 0, 2, 3, "opposite", "rejects", false, false], [2, 3, 6, 7, "opposite", "calls_for_boycot_of", false, false], [2, 3, 6, 7, "topic", "", false, false]]}
```

Problems:

+ The news domain is CoNLL-03 text (the example above is the first CoNLL-03 train sentence). This overlaps with CleanCoNLL.
+ The design says "Confirm whether every split has at least one relation per sentence". Answer: no. Relation-free sentences exist in every domain and split.

## 5. Hallmarks of Cancer (multi-label classification)

| Field | Value |
|---|---|
| Source | HF `bigbio/hallmarks_of_cancer` parquet ref `b78c5a2c56ed30bb64f184b204c46362092cebdf` (main `5177d3fb0681f27af37431f46617fea31d50bdc3` is script-only) |
| Configs | `hallmarks_of_cancer_source` (int labels) and `hallmarks_of_cancer_bigbio_text` (string labels). Same rows. |
| File sha256 (bigbio_text) | train `9ec46bb3b489ab70e6882b4db2b1b654e7877bc08dae59797ab11d66527e9642`, validation `e4ecc79b353260fc462334c78189c4c4ffa93aafe2797d2c376fbf054098fbe1`, test `a5c79c5c52d7990dc7c1dcba22c1323c29c503948db17aabe233e5ec26b7698f` |
| Upstream check | The script reads `sb895/Hallmarks-of-Cancer` master and BLURB split files. Upstream head is `3bc6e6deb90cf5bc4f9988c493fe2c32e9b3cf2a` (2018-09-18), so the conversion cannot have drifted since. I fetched it (zip sha256 `80a5ba6ada6b722158f3664d33844e5090918ee5e68f7c46da141727ad78a74a`) and BLURB `data_generation.tar.gz` (sha256 `5f749cbaa2b137aa1086d5610ef8dc5e182cc2a523173e6f26e394c0efad5898`). All 1,852 documents have equal sentence and label line counts, so the script's `zip` pairing does not shift labels. |
| Licence | GPL-3.0 (upstream GitHub licence API, BigBio card, and script `_LICENSE`) |
| Splits | train, validation, test (BLURB PMID lists 1,295 / 186 / 371). Dev exists. |
| Documents | 1,295 / 186 / 371 PMIDs; 0 overlap between splits |
| Sentences | 12,119 / 1,798 / 3,547 |
| `document_id` | `<PMID>_<sentence index>`, for example `22239943_0`. Strip the suffix to get the PMID. |

Label counts (sentence-level; a sentence can have several):

| Label | Train | Validation | Test |
|---|---|---|---|
| none | 9,027 | 1,334 | 2,649 |
| sustaining proliferative signaling | 674 | 85 | 234 |
| resisting cell death | 602 | 72 | 158 |
| genomic instability and mutation | 523 | 103 | 142 |
| activating invasion and metastasis | 448 | 64 | 155 |
| tumor promoting inflammation | 375 | 45 | 98 |
| evading growth suppressors | 268 | 23 | 75 |
| inducing angiogenesis | 238 | 60 | 60 |
| enabling replicative immortality | 219 | 11 | 65 |
| avoiding immune destruction | 185 | 20 | 21 |
| cellular energetics | 136 | 44 | 33 |

Labels per sentence (train): 0 labels 7, 1 label 11,592, 2 labels 464, 3 labels 50, 4 labels 5, 5 labels 1.
"none" never co-occurs with another label.

Record format (`hallmarks_of_cancer_bigbio_text`, train row 0):

```
{"id": "1", "document_id": "22239943_0", "text": "Hypoxic events frequently occur in the aquatic environment in association with micro pollutants , including heavy metals .", "labels": ["none"]}
```

Problems:

+ 11 rows (7 / 3 / 1) have an empty label list, not "none". The upstream label was only `NULL`, which the script drops. Map them to the empty set, like "none".
+ Some text is cut, for example `22888341_8`: "The intratumoral microvessel density was h". This is in the upstream text (unconfirmed how many).
+ Text is pre-tokenised with spaces around punctuation.
+ Duplicate texts: 7 / 0 / 4.

## 6. MASSIVE 1.1 en-US (slots)

| Field | Value |
|---|---|
| Source | `https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz` (link in `alexa/massive` README, commit `f966f21846043aabef9b0f974fa7970027f43738`) |
| Tarball sha256 | `4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577` (40,251,390 bytes) |
| File | `1.1/data/en-US.jsonl`, sha256 `c70f75c6a543a26e249ec383df67733ad9b1066f6c0406c2e04a3f03356e407e` |
| Licence | CC BY 4.0 (`1.1/LICENSE`, "Attribution 4.0 International"); `NOTICE.md`: derived from SLURP, also CC BY 4.0 |
| Splits | `partition` field: train, dev, test. Dev exists. |
| Utterances | 11,514 / 2,033 / 2,974 (16,521 total, all ids unique) |
| No-slot utterances | 3,755 (32.6%) / 646 (31.8%) / 994 (33.4%) |
| Slot mentions | 11,344 / 2,012 / 2,815 |
| Slot types | 55 in train, 53 in dev, 53 in test |
| Intents | 60 in train, 59 in dev and test |
| Utterances with a repeated slot type | 361 / 67 / 77 |
| Duplicate utterance texts | 46 / 2 / 4 |

Slot format: `annot_utt` marks each slot inline as `[slot_type : value]`.
I parsed it with the regex `\[(\S+?) : (.*?)\]`.
Removing the brackets gives back `utt` exactly in every row (0 mismatches), so every value is an exact substring with known character offsets.

Top train slot types: date 1,816, place_name 1,086, event_name 998, person 864, time 797, media_type 477, business_name 383, weather_descriptor 320, transport_type 318, food_type 295.
Rarest: music_album 1, game_type 1, sport_type 5, audiobook_author 6.

Record format (`en-US.jsonl`, line 1):

```
{"id": "0", "locale": "en-US", "partition": "test", "scenario": "alarm", "intent": "alarm_set", "utt": "wake me up at five am this week", "annot_utt": "wake me up at [time : five am] [date : this week]", "worker_id": "1"}
```

## 7. MASSIVE 1.1 fr-FR (slots)

| Field | Value |
|---|---|
| Source | same tarball as en-US |
| File | `1.1/data/fr-FR.jsonl`, sha256 `f9bf3db170ad415b389e4c9594dd0f8f80c38188143e05cc4459a6fa7df7cf49` |
| Licence | CC BY 4.0 |
| Utterances | 11,514 / 2,033 / 2,974 |
| Shared ids with en-US | yes: the id sets are equal (16,521) |
| Same partition per id | yes, for all 16,521 ids |
| Same intent per id | yes, for all ids |
| No-slot utterances | 3,927 (34.1%) / 684 (33.6%) / 1,045 (35.1%) |
| Slot mentions | 11,100 / 1,958 / 2,744 |
| Slot type multiset differs from en-US | 433 ids |
| `slot_method` counts | translation 10,091, unchanged 3,062, localization 1,705, unchanged_translation 809, unchanged_localization 23 |
| Duplicate utterance texts | 325 / 17 / 31 |
| `annot_utt` vs `utt` | 0 mismatches after bracket removal |

Record format (`fr-FR.jsonl`, line 1, shortened):

```
{"id": "0", "locale": "fr-FR", "partition": "test", "scenario": "alarm", "intent": "alarm_set", "utt": "réveille-moi à cinq heures du matin cette semaine", "annot_utt": "réveille-moi à [time : cinq heures du matin] [date : cette semaine]", "worker_id": "22", "slot_method": [{"slot": "time", "method": "translation"}, {"slot": "date", "method": "translation"}], "judgments": [{"worker_id": "22", "intent_score": 1, "slots_score": 1, "grammar_score": 4, "spelling_score": 2, "language_identification": "target"}, ...]}
```

Problems:

+ Translation drops slots: fr-FR has 172 more no-slot train utterances than en-US.
+ fr-FR has far more duplicate texts in train (325 vs 46): different English requests collapse to one French string.
+ The fr-FR rows carry `judgments` (per-worker quality scores). They could support the French slot-quality check; the pool is never filtered on them (design section 13).

## Blockers and open points

1. CleanCoNLL text licence - the build uses the unofficial deepai mirror of Reuters text. Decide if that is acceptable, or get CoNLL-03 through the official route. Never release the built text.
2. CleanCoNLL annotation licence - no licence file in the repo (unconfirmed terms).
3. MIT Movie licence - `unknown` on the Galileo card. Release gate stays.
4. BC5CDR sentence quality - decide: keep TNER sentences (bad decimal splits, 6 rows with lost words) or re-split from PubTator.
5. MIT Movie thesis link - the repo JSON is the original MIT data with dev = test. The new runs use Galileo; the thesis comparison must say so.

## Changelog

- 2026-09-25 12:30 CEST - Created.
