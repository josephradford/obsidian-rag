# Data Provenance

## Source
- **Type:** Personal Obsidian vault (Markdown files)
- **Location:** Local filesystem, synced via iCloud
- **Access:** Private — not included in repository
- **Content:** Personal notes on [topics — fill in]
- **Volume:** [X files, ~Y MB — fill in after first ingest]

## Preprocessing
- File types: .md and .pdf (configurable via FILE_EXTENSIONS)
- Excluded: .obsidian/, .trash/, templates/
- Preprocessing applied: [fill in — e.g. wikilink stripping, frontmatter handling, PDF text extraction]

## Versioning approach
- Raw vault is not versioned (personal, changes constantly)
- Ingestion metadata tracked via MLflow (file count, chunk count, timestamp)
- Evaluation dataset (eval/test_questions.json) IS version controlled
