# Dataset

MOSS-Conversations-Dataset (nur Englisch) für Benchmarking.

## Datei

`moss-003-sft-nochinese.jsonl` - ~300k englische Q&A-Paare

## Quelle

Original: https://github.com/OpenLMLab/MOSS (gefiltert für Englisch)

## Format

JSONL mit Conversation-Struktur:

```json
{
    "chat": {
        "turn_1": {
            "Human": "<query>",
            "MOSS": "<response>"
        }
    }
}
```
