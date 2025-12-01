# HWU Dataset Description

This directory contains metadata and descriptions of the dataset used in this project.
**Note:** The actual dataset files are not included in the repository to avoid overloading storage/token limits as per project requirements.

## Dataset Structure

The dataset consists of several CSV files containing labeled text data for intent classification (NLU) tasks, and a JSON file listing the categories.

### Files Overview

| File Name | Description | Rows | Columns |
|-----------|-------------|------|---------|
| `train.csv` | Full training dataset | 8954 | `text`, `category` |
| `val.csv` | Validation dataset | 1076 | `text`, `category` |
| `test.csv` | Testing dataset | 1076 | `text`, `category` |
| `train_5.csv` | Small subset of training data (few-shot 5 samples) | 320 | `text`, `category` |
| `train_10.csv`| Small subset of training data (few-shot 10 samples)| 640 | `text`, `category` |
| `categories.json` | List of all intent categories | 64 items | N/A |

### Data Schema

**CSV Files (`train.csv`, `val.csv`, etc.)**
- **text**: The input utterance or command (String).
- **category**: The intent label associated with the text (String).

**Example Data (from `train.csv`)**
| text | category |
|------|----------|
| "what alarms do i have set right now" | `alarm_query` |
| "checkout today alarm of meeting" | `alarm_query` |
| "report alarm settings" | `alarm_query` |

### Categories Sample
The `categories.json` file contains 64 intents, covering domains like:
- **Alarm**: `alarm_query`, `alarm_remove`, `alarm_set`
- **Audio**: `audio_volume_down`, `audio_volume_mute`, `audio_volume_up`
- **IoT**: `iot_cleaning`, `iot_coffee`, `iot_hue_lightchange`
- **General**: `general_affirm`, `general_joke`, `general_quirky`

---
## Additional Datasets (Update Phase 2)

Besides the HWU intent dataset, the project also utilizes unstructured text data for pre-training or word embedding tasks.

### Files Overview

| File Name | Type | Lines | Description |
|-----------|------|-------|-------------|
| `en_ewt-ud-train.txt` | Plain Text | 7 | Simple sentences for basic NLP (Word Analogy, Tokenization). Contains classic examples like "quick brown fox" and "king/queen". |
| `c4-train...json` | JSON Lines | 6 | Tech-focused corpus (Apache Spark, Hardware). Each line is a JSON object with a "text" field. |

### Data Schema & Content

#### 1. `en_ewt-ud-train.txt` (Plain Text)
* **Format:** Raw sentences separated by newlines.
* **Purpose:** Suitable for testing basic tokenizer or Word2Vec relationships (e.g., King - Man + Woman = Queen).
* **Sample:**
    > the quick brown fox jumps over the lazy dog
    > a king is a powerful man
    > a queen is a powerful woman

#### 2. `c4-train.00000-of-01024-30K.json` (JSONL)
* **Format:** JSON Lines. Each line is a valid JSON object.
* **Schema:** `{"text": "String content..."}`
* **Content:** Technical definitions related to Big Data (Spark) and Computer Hardware.
* **Sample:**
    ```json
    {"text": "Apache Spark is a unified analytics engine for large-scale data processing."}
    {"text": "A modern personal computer contains a CPU, RAM, and a storage device."}
    ```