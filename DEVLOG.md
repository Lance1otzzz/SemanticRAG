# Development Log

This file briefly tracks modifications made to the repository.

## Initial implementation

- Added simple fallback implementations for chunking and embedding so the
  project can run without the optional external libraries.
- `ChonkieTextChunker` now records paragraph and offset metadata when the
  `chonkie` library is not available.
- `TextEmbedder` supports a hashing based embedding for offline tests and can
  save embedding records as JSONL.
- Created small unit tests in `tests/` covering both modules.
- Updated `README.md` with setup instructions and project overview.

## Code cleanup

- Removed example manual test code from `ChonkieTextChunker` to improve code cleanliness

## Bug fixes

- Fixed Chonkie library integration in `ChonkieTextChunker`:
  - Updated import statements to use `RecursiveChunker` instead of `Chonkie` and `DefaultChunker`
  - Updated the chunking API calls to match the latest Chonkie library interface
  - Improved metadata handling to properly merge user-provided metadata with chunk metadata
  - Added proper fallback implementation when Chonkie is not available
