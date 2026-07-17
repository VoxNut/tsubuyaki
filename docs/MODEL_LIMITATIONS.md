# Model limitations

The model is a contextual reading classifier for known ambiguous Kanji
surfaces. It is not an end-to-end Furigana generator for arbitrary Japanese.

- Contextual BERT/MLP inference covers 1,004 ambiguous surfaces reconstructed
  from the training split.
- The broader known-surface map contains 26,210 surfaces and uses the
  most-frequent training reading when a contextual candidate set is not
  available.
- Unseen surfaces use MeCab. If MeCab provides no valid Kana reading, the
  service preserves the original text as `PlainText`.
- MeCab and Most-Frequent results do not receive invented confidence scores.
- The model was evaluated on a group split by `file_path`; results must not be
  presented as accuracy on arbitrary modern dialogue, names, domain terms, or
  every unseen Kanji compound.
- The production target-centered window intentionally improves long-input
  handling compared with the notebook's original right truncation. This keeps
  both target markers and the complete target but can change logits for very
  long sentences.
- Automatic readings can be wrong. The Phase 2 frontend must preserve manual
  edits and make prediction source/confidence visible.

The completed test report selected `HybridTuned` on validation and reported
90.60% exact match with 8.78% CER on the held-out test split. Those figures
describe that fixed experiment, not a guarantee for uploaded subtitles.
