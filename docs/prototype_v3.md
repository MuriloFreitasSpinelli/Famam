# FAMAM v3 Prototype Documentation

## Overview

The v3 prototype was an early implementation of the FAMAM music generation system. It established the core pipeline for processing MIDI files, encoding musical data, and training autoregressive models. While functional, several issues were identified that led to the development of v4.

## Architecture

### Data Pipeline
```
MIDI Files → Preprocessing → Encoding → Dataset → Training → Generation
```

### Components
- **Preprocessing**: Resolution adjustment, quantization, track filtering
- **Encoders**: Event-based tokenization of musical sequences
- **Models**: Transformer and LSTM architectures for next-token prediction
- **Dataset**: HDF5-based storage with vocabulary management

## Issues Encountered

### 1. Segmentation Problems

**Problem**: Fixed-length segmentation in ticks did not align with musical bar boundaries, causing segments to cut through measures unpredictably.

```
Before (v3): segment_length = 1024 ticks (arbitrary)
             → Segments could start/end mid-bar
             → Loss of musical coherence at segment boundaries
```

**Impact**: Generated music often had abrupt transitions and lost rhythmic structure at segment boundaries.

### 2. Genre Vocabulary Not Registered

**Problem**: When adding entries to the dataset, genres were stored in entries but never registered in the vocabulary mapping.

```python
# v3 bug: genre stored but not registered
self.entries.append(MusicEntry(music=music, genre=genre, ...))
# vocabulary.genre_to_id remained empty
```

**Impact**: Models trained without genre conditioning. The dataset reported 0 genres despite having genre labels.

### 3. Single-Track Encoding Limitation

**Problem**: The initial encoder processed tracks independently, losing the relationship between instruments playing simultaneously.

**Impact**: Generated outputs lacked the interplay between instruments (e.g., bass following chord changes, drums syncing with guitar).

### 4. Padding Ratio Issues

**Problem**: Segments with too much padding (empty space) were included in training, teaching the model to generate silence.

**Impact**: Some generated sequences contained long stretches of no events.

### 5. Transposition Augmentation Edge Cases

**Problem**: Drum tracks were incorrectly transposed along with melodic tracks.

**Impact**: Drum patterns became musically invalid (e.g., snare mapped to hi-hat pitch).

## Lessons Learned

1. **Bar-aligned segmentation** preserves musical structure better than arbitrary tick counts
2. **Vocabulary registration** must happen at data insertion time, not just at encoding time
3. **Multi-track encoding** is essential for learning instrument relationships
4. **Augmentation** requires instrument-type awareness (drums vs melodic)

## Migration to v4

The v4 implementation addresses these issues with:

- `segment_bars` parameter for bar-aligned segmentation
- Proper genre registration in `MusicDataset.add()`
- `MultiTrackEncoder` for full-song encoding with all instruments
- Drum-aware transposition (drums excluded from pitch shifting)
- Configurable `max_padding_ratio` to filter sparse segments

---

*This document summarizes the v3 prototype development and the issues that informed v4 design decisions.*
