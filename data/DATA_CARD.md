# Data Card

## Source

**Salesforce xLAM Function Calling 60K**
- Repository: `Salesforce/xlam-function-calling-60k`
- License: **CC-BY-NC-4.0** (Creative Commons Attribution-NonCommercial)
- Original size: 60,000 examples

## License Notice

This data is derived from the xLAM dataset and inherits its **CC-BY-NC-4.0** license.

**Research use only. Not for commercial applications.**

## Processing Pipeline

```
xLAM 60K (Salesforce)
    ↓
Filter to top 50 functions → 23,106 examples
    ↓
Remove problematic functions (4) → schema inconsistencies
    ↓
Remove multi-call examples → single function calls only
    ↓
Convert to probe format → character spans
    ↓
Fix remaining issues → 9,860 clean examples
    ↓
Shuffle (seed=42) → compiled_clean_shuffled.jsonl
```

## Files

### compiled_clean_shuffled.jsonl

**9,860 examples** in JSONL format.

```json
{
  "function": "book_restaurant",
  "utterance": "Book a table for 4 at 7pm",
  "slots": {
    "party_size": {"char_start": 18, "char_end": 19, "value": "4"},
    "time": {"char_start": 23, "char_end": 26, "value": "7pm"}
  }
}
```

Fields:
- `function`: Function name (45 unique functions)
- `utterance`: Natural language user input
- `slots`: Dictionary of present slots with character spans
  - `char_start`: Start character offset (inclusive)
  - `char_end`: End character offset (exclusive)
  - `value`: The extracted string

### function_schemas.json

**45 functions** with slot definitions.

```json
{
  "book_restaurant": {
    "slots": {
      "party_size": {"type": "integer"},
      "time": {"type": "string"},
      "restaurant_name": {"type": "string"},
      "date": {"type": "string"}
    }
  },
  ...
}
```

## Split

Pre-shuffled with `random.seed(42)` for reproducibility.

| Split | Lines | Count |
|-------|-------|-------|
| Train | 0-8873 | 8,874 (90%) |
| Holdout | 8874-9859 | 986 (10%) |

To split:
```python
with open("compiled_clean_shuffled.jsonl") as f:
    lines = f.readlines()
train = lines[:8874]
holdout = lines[8874:]
```

## Excluded Functions

Four functions were removed due to schema inconsistencies:

| Function | Issue |
|----------|-------|
| search | 60+ different slot names across examples |
| loginuser | Extra `toolbench_rapidapi_key` slot |
| sort_numbers | Uses `nums` but schema says `numbers` |
| auto_complete | Multiple APIs with different slots |

## Statistics

- **Functions**: 45
- **Total slots across all functions**: 99
- **Examples**: 9,860
- **Average slots per function**: 2.2
- **Average slots filled per example**: 1.8

## Character Span Convention

Spans use Python string slicing convention:
- `char_start`: inclusive
- `char_end`: exclusive
- `utterance[char_start:char_end] == value`

Example:
```python
utterance = "Book a table for 4 at 7pm"
#            0         1         2
#            0123456789012345678901234567

# party_size: char_start=18, char_end=19
utterance[18:19]  # "4"

# time: char_start=23, char_end=26
utterance[23:26]  # "7pm"
```

## Citation

```
@dataset{xlam2024,
  title={xLAM Function Calling 60K},
  author={Salesforce AI Research},
  year={2024},
  url={https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k}
}
```
