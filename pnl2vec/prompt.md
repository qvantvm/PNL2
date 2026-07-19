You are an expert machine-learning engineer specializing in symbolic music, tokenization, representation learning, PyTorch, and visualization.

Build a complete, well-tested Python project called:

pnl2vec

The project must:

1. Parse a corpus of valid PNL/2 piano-notation documents.
2. Tokenize PNL/2 without introducing syntactic ambiguity.
3. Train token embeddings from scratch using musical context.
4. Visualize and inspect the learned embedding space.
5. Quantitatively evaluate whether the embeddings capture meaningful musical relationships.
6. Export the tokenizer, vocabulary, embeddings, metadata, and analysis results.
7. Provide a simple Python API and CLI for using the trained representations in future PNL language models, classifiers, retrieval systems, and music-analysis tools.

Do not implement a large transformer in the first version. The purpose of this project is to understand and validate the tokenizer and learned embedding space before introducing a more complex model.

⸻

1. Core principles

The implementation must follow these principles:

* PNL/2 syntax must be parsed structurally, not split naïvely on whitespace.
* Tokenization must be deterministic and reversible where practical.
* No token string may have two unrelated meanings.
* Structural, musical, and numeric information must remain distinguishable.
* The model must learn embeddings through a real training objective.
* Do not claim that a randomly initialized nn.Embedding layer contains semantic information.
* Separate lexical token identity from optional musical attributes.
* Preserve document, part, measure, voice, hand, and temporal boundaries.
* Prevent context windows from crossing unrelated documents.
* Make all experiments reproducible through explicit random seeds.
* Keep the system modular so that the embedding model can later be replaced by a transformer.

Use Python 3.11 or newer and PyTorch.

⸻

2. Repository structure

Create the following structure:

pnl2vec/
├── README.md
├── pyproject.toml
├── configs/
│   ├── tokenizer.yaml
│   ├── train_skipgram.yaml
│   ├── train_cbow.yaml
│   └── visualization.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── examples/
├── artifacts/
│   ├── tokenizer/
│   ├── checkpoints/
│   ├── embeddings/
│   ├── visualizations/
│   └── reports/
├── src/
│   └── pnl2vec/
│       ├── __init__.py
│       ├── pnl/
│       │   ├── lexer.py
│       │   ├── parser.py
│       │   ├── ast.py
│       │   ├── validation.py
│       │   └── serializer.py
│       ├── tokenizer/
│       │   ├── token.py
│       │   ├── vocabulary.py
│       │   ├── tokenizer.py
│       │   ├── normalization.py
│       │   └── serialization.py
│       ├── corpus/
│       │   ├── loader.py
│       │   ├── examples.py
│       │   ├── statistics.py
│       │   └── split.py
│       ├── models/
│       │   ├── skipgram.py
│       │   ├── cbow.py
│       │   ├── negative_sampling.py
│       │   └── losses.py
│       ├── training/
│       │   ├── dataset.py
│       │   ├── trainer.py
│       │   ├── checkpoint.py
│       │   └── seed.py
│       ├── evaluation/
│       │   ├── similarity.py
│       │   ├── analogies.py
│       │   ├── probes.py
│       │   ├── retrieval.py
│       │   └── report.py
│       ├── visualization/
│       │   ├── projection.py
│       │   ├── plots.py
│       │   └── interactive.py
│       ├── api.py
│       └── cli.py
├── scripts/
│   ├── generate_synthetic_corpus.py
│   ├── inspect_corpus.py
│   ├── train.py
│   ├── evaluate.py
│   ├── visualize.py
│   └── query_embeddings.py
└── tests/
    ├── test_lexer.py
    ├── test_parser.py
    ├── test_roundtrip.py
    ├── test_tokenizer.py
    ├── test_vocabulary.py
    ├── test_training_examples.py
    ├── test_model.py
    ├── test_evaluation.py
    └── test_cli.py

Use a src/ layout and make the package installable with:

pip install -e ".[dev]"

⸻

3. PNL/2 grammar integration

The repository may already contain the complete PNL/2 grammar. Search the repository before implementing a substitute.

If a canonical grammar or parser exists:

* reuse it;
* do not create a conflicting PNL dialect;
* adapt its AST into the tokenizer;
* document all assumptions.

If the grammar is not available, isolate the grammar-dependent implementation behind typed interfaces and implement a clearly labeled minimal PNL/2 subset sufficient for the examples and tests.

The minimal fallback parser must recognize at least:

* version header;
* metadata;
* parts;
* measures;
* voices or layers;
* hand assignment;
* notes;
* rests;
* chords or simultaneous events;
* pitch spelling;
* octave;
* duration;
* dots;
* ties;
* slurs;
* articulation;
* dynamics;
* pedal events;
* fingering;
* barlines;
* tempo;
* key;
* time signature;
* comments.

Do not silently accept malformed syntax. Return errors containing:

* filename;
* line number;
* column number;
* unexpected symbol;
* expected construct;
* short contextual excerpt.

Create a typed AST using dataclasses or equivalent typed structures.

Provide:

parse_pnl(text: str) -> Document
serialize_pnl(document: Document) -> str
validate_pnl(document: Document) -> list[ValidationIssue]

Where applicable, verify this round trip:

parse_pnl(serialize_pnl(parse_pnl(source)))

must produce an equivalent AST.

⸻

4. Token representation

Define an explicit token object:

@dataclass(frozen=True)
class Token:
    kind: TokenKind
    value: str | int | float | None
    attributes: Mapping[str, str | int | float | bool]
    source_span: SourceSpan | None

Each token must have a unique canonical serialized form.

Examples of conceptual token categories include:

SPECIAL:BOS
SPECIAL:EOS
SPECIAL:PAD
SPECIAL:UNK
STRUCT:DOC_START
STRUCT:DOC_END
STRUCT:PART_START
STRUCT:PART_END
STRUCT:MEASURE_START
STRUCT:MEASURE_END
STRUCT:VOICE_START
STRUCT:VOICE_END
STRUCT:CHORD_START
STRUCT:CHORD_END
HAND:LEFT
HAND:RIGHT
PITCH_CLASS:C
ACCIDENTAL:NATURAL
OCTAVE:4
DURATION:1/4
DOT_COUNT:1
ARTICULATION:STACCATO
DYNAMIC:MF
SLUR:START
SLUR:END
TIE:START
TIE:END
PEDAL:DOWN
PEDAL:UP
FINGER:3
REST
BARLINE:DOUBLE
KEY:D_MAJOR
TIME_SIGNATURE:4/4
TEMPO_BPM:120

These are examples, not permission to overwrite the canonical PNL/2 syntax.

Do not use the same surface token for:

* pitch and key;
* duration and numeric metadata;
* fingering and octave;
* voice number and arbitrary integer;
* slur and tie;
* pedal and note duration.

Token identity must encode the semantic namespace.

⸻

5. Tokenization strategies

Implement at least two tokenizer modes.

5.1 Atomic structural tokenizer

Decompose an event into meaningful atomic tokens.

For example, a note may produce conceptual components such as:

EVENT:NOTE
PITCH_CLASS:C
ACCIDENTAL:SHARP
OCTAVE:4
DURATION:1/8
ARTICULATION:ACCENT
FINGER:3

Advantages:

* small vocabulary;
* compositional;
* generalizes to unseen combinations.

5.2 Compound event tokenizer

Represent frequent complete musical events as compound tokens, such as:

NOTE:C#4:DUR_1/8

Keep less common expressive properties as separate tokens.

Advantages:

* shorter sequences;
* stronger local event identity.

The compound tokenizer must use a configurable frequency threshold. Rare compounds must fall back to atomic tokenization rather than becoming UNK.

Expose the mode in configuration:

tokenizer:
  mode: atomic

or:

tokenizer:
  mode: compound
  compound_min_frequency: 20

⸻

6. Vocabulary

Implement a vocabulary class supporting:

token_to_id(token: str) -> int
id_to_token(token_id: int) -> str
encode(tokens: Sequence[Token]) -> list[int]
decode(ids: Sequence[int]) -> list[Token]
save(path: Path) -> None
load(path: Path) -> Vocabulary

Reserve stable IDs for:

<PAD>
<UNK>
<BOS>
<EOS>
<MASK>
<DOC_SEP>
<MEASURE_SEP>

The vocabulary must be built only from the training split. Validation and test data must not influence it.

Write:

artifacts/tokenizer/vocabulary.json
artifacts/tokenizer/token_metadata.json
artifacts/tokenizer/config.yaml

Include for each vocabulary item:

* ID;
* canonical string;
* token kind;
* token value;
* musical category;
* training frequency;
* whether it is special;
* whether it is atomic or compound.

⸻

7. Corpus preparation

Support reading:

data/raw/**/*.pnl

Split the corpus by complete musical work or document, never by arbitrary token window.

Default split:

* 80% training;
* 10% validation;
* 10% test.

Use a deterministic seed.

Create a corpus report containing:

* number of documents;
* number of measures;
* number of events;
* number of tokens;
* vocabulary size;
* out-of-vocabulary rate;
* token frequency distribution;
* average sequence length;
* pitch-class distribution;
* duration distribution;
* articulation frequency;
* dynamic frequency;
* pedal-event frequency;
* fingering frequency;
* parse failures and validation warnings.

Save the report as both JSON and Markdown.

⸻

8. Synthetic corpus generator

Because a real PNL/2 corpus may initially be small, implement a synthetic corpus generator.

It must generate valid, musically structured PNL/2 rather than random token noise.

Generate examples in multiple keys, meters, registers, and hands, including:

* major and minor scales;
* diatonic triads;
* seventh chords;
* inversions;
* arpeggios;
* cadences;
* common harmonic progressions;
* transposed motifs;
* rhythmic variations;
* repeated accompaniment patterns;
* Alberti bass;
* left-hand chord patterns;
* simple counterpoint;
* slurs and phrase boundaries;
* accents and staccato;
* dynamic changes;
* sustain-pedal spans;
* fingerings;
* rests;
* ties;
* multiple voices;
* chords;
* hand crossings in a limited number of examples.

Avoid encoding only trivial correlations. For example:

* do not always assign low notes to the left hand;
* do not always use pedal with chords;
* do not always use one fingering for one pitch;
* do not make every dominant chord immediately followed by tonic;
* include controlled negative and contrasting examples.

Make the generator deterministic under a seed.

Clearly mark synthetic documents in metadata.

Generate at least three configurable corpus sizes:

tiny:   approximately 100 documents
small:  approximately 1,000 documents
medium: approximately 10,000 documents

Do not automatically generate the medium corpus during tests.

⸻

9. Training objectives

Implement two classical embedding-training objectives.

9.1 Skip-gram with negative sampling

Given a center token, predict nearby context tokens.

Implement the model without using a pre-trained embedding package.

Required components:

input_embeddings = nn.Embedding(vocab_size, embedding_dim)
output_embeddings = nn.Embedding(vocab_size, embedding_dim)

Use negative sampling with a configurable unigram distribution raised to the power of 0.75.

Support dynamic context windows.

Exclude padding and document-boundary tokens from negative samples.

9.2 CBOW with negative sampling

Given surrounding context tokens, predict the center token.

Implement this independently enough that both objectives can be compared.

⸻

10. Music-aware context policies

A naïve linear text window is not always appropriate for music. Implement configurable context policies:

A. Linear token window

Use nearby serialized tokens.

B. Event window

Use nearby musical events, regardless of how many atomic tokens compose each event.

C. Measure-local context

Sample context from the same measure.

D. Voice-local context

Sample context from the same voice.

E. Temporally aligned context

Include simultaneous events in other voices or hands.

F. Hybrid context

Combine:

* nearby events in the same voice;
* simultaneous events;
* adjacent harmonic events;
* measure and phrase markers.

Each positive pair should optionally carry a relation label such as:

LINEAR_NEIGHBOR
SAME_EVENT
NEXT_EVENT
PREVIOUS_EVENT
SIMULTANEOUS
SAME_MEASURE
SAME_VOICE
OTHER_HAND
PHRASE_NEIGHBOR

The first training model may ignore the relation label, but the dataset must retain it for analysis and future relation-aware models.

Add configurable sampling weights for these relation types.

Default to an event-aware hybrid policy rather than a raw token-only window.

⸻

11. Prevent trivial embedding behavior

Atomic tokens belonging to one event naturally appear adjacent. This may cause the model to learn only that OCTAVE:4 occurs beside PITCH_CLASS:C.

Address this explicitly.

Implement configuration options to:

* include or exclude same-event pairs;
* weight same-event context separately;
* sample next-event and simultaneous-event context;
* cap the proportion of same-event pairs;
* compare embeddings trained with and without same-event context.

Default:

context:
  same_event_weight: 0.25
  sequential_event_weight: 1.0
  simultaneous_event_weight: 1.0
  same_measure_weight: 0.5

Document why this matters.

⸻

12. Training configuration

Provide YAML configuration with fields such as:

seed: 42
data:
  raw_dir: data/raw
  processed_dir: data/processed
  split_by: document
tokenizer:
  mode: atomic
  normalize_enharmonics: false
  preserve_pitch_spelling: true
  preserve_source_spans: true
model:
  objective: skipgram
  embedding_dim: 128
  sparse_embeddings: false
context:
  policy: hybrid
  min_window: 1
  max_window: 4
  include_same_event: true
  same_event_weight: 0.25
  sequential_event_weight: 1.0
  simultaneous_event_weight: 1.0
  same_measure_weight: 0.5
training:
  epochs: 20
  batch_size: 1024
  learning_rate: 0.003
  negative_samples: 10
  optimizer: adamw
  weight_decay: 0.0001
  gradient_clip_norm: 1.0
  early_stopping_patience: 4
  device: auto
  num_workers: 0
logging:
  log_every_steps: 100
  save_every_epochs: 1

Support CPU, CUDA, and Apple Silicon MPS.

The project must run correctly on macOS with an Apple M-series processor.

Select the device in this order:

1. CUDA, when available;
2. MPS, when available;
3. CPU.

Do not assume MPS supports every sparse operation. Use safe defaults.

⸻

13. Training outputs

Save:

artifacts/checkpoints/best.pt
artifacts/checkpoints/latest.pt
artifacts/embeddings/input_embeddings.npy
artifacts/embeddings/output_embeddings.npy
artifacts/embeddings/combined_embeddings.npy
artifacts/embeddings/token_ids.json
artifacts/embeddings/metadata.json
artifacts/reports/training_history.json
artifacts/reports/training_report.md

For skip-gram, expose three embedding choices:

* input embedding;
* output embedding;
* normalized average of input and output embeddings.

Use the averaged representation as the default for inspection, but allow the user to select another.

Track:

* training loss;
* validation loss;
* positive-pair score;
* negative-pair score;
* learning rate;
* epoch duration;
* nearest-neighbor stability between checkpoints.

⸻

14. Visualization

Implement:

* PCA;
* t-SNE;
* UMAP when the optional dependency is installed.

Never run t-SNE or UMAP over an enormous vocabulary without sampling. Provide configurable frequency and category filters.

Generate static plots using Matplotlib and interactive HTML using Plotly.

Required visualizations:

1. All major token categories.
2. Pitch-class tokens.
3. Pitches grouped by octave.
4. Duration tokens.
5. Dynamics.
6. Articulations.
7. Fingering tokens.
8. Pedal tokens.
9. Structural tokens.
10. Chord tokens, when available.
11. High-frequency nearest-neighbor graph.
12. Comparison of embeddings before and after training.

Interactive points must show hover metadata:

* token string;
* token ID;
* kind;
* value;
* frequency;
* nearest neighbors;
* optional musical attributes.

Use shape or marker style as well as color where practical so the visualization remains interpretable for color-vision deficiencies.

Save to:

artifacts/visualizations/

⸻

15. Nearest-neighbor inspection

Implement cosine-similarity search.

CLI examples:

pnl2vec neighbors "PITCH_CLASS:C" --top-k 12
pnl2vec neighbors "DURATION:1/4" --top-k 12
pnl2vec neighbors "PEDAL:DOWN" --top-k 12

Support category filtering:

pnl2vec neighbors "PITCH_CLASS:C" \
  --category pitch \
  --top-k 12

Return:

* neighbor token;
* cosine similarity;
* frequency;
* category;
* relation summary from the corpus.

Do not allow the queried token itself to appear as its own neighbor.

⸻

16. Phrase and document embeddings

Token embeddings alone do not directly represent a complete phrase. Implement simple downstream aggregation methods:

embed_tokens(ids, pooling="mean")
embed_pnl(text, pooling="mean")
embed_events(events, pooling="mean")

Support:

* mean pooling;
* frequency-weighted mean;
* inverse-frequency-weighted mean;
* smooth inverse frequency;
* mean with removal of the first principal component.

Ignore structural and special tokens by default during phrase pooling, but make this configurable.

Also implement event embeddings by combining the atomic tokens belonging to one musical event.

Document clearly that these pooled vectors are baselines, not full contextual sequence representations.

⸻

17. Retrieval demonstration

Create a small semantic retrieval demonstration.

Given a PNL/2 query phrase:

1. tokenize it;
2. compute a pooled phrase embedding;
3. search a database of PNL phrases using cosine similarity;
4. return the most similar phrases;
5. show both the source PNL and a compact human-readable musical summary.

CLI:

pnl2vec index data/raw --output artifacts/phrase_index
pnl2vec search query.pnl --index artifacts/phrase_index --top-k 10

The index may use NumPy brute-force search initially. Keep the API abstract enough to add FAISS later.

⸻

18. Evaluation

Do not judge embedding quality only from a two-dimensional plot.

Implement the following evaluations.

18.1 Intrinsic nearest-neighbor tests

Create expected-neighbor sets such as:

* adjacent duration values;
* the same pitch class in nearby octaves;
* enharmonic spellings, when normalization is enabled;
* related dynamics;
* articulation pairs;
* pedal down/up relationships;
* structurally related boundary tokens.

Report precision@k and mean reciprocal rank where an expected set is meaningful.

18.2 Music-theory similarity tests

For pitch and chord representations, test whether embeddings reflect:

* octave equivalence;
* pitch-class identity;
* interval proximity;
* circle-of-fifths relationships;
* chord-root similarity;
* chord-quality similarity;
* inversion relationships;
* relative major/minor relationships.

Do not assert that all these relationships must emerge from every corpus. Report measured results and limitations.

18.3 Analogy tests

Support vector arithmetic experiments such as:

PITCH:C4 - OCTAVE:4 + OCTAVE:5 ≈ PITCH:C5
CHORD:C_MAJOR - ROOT:C + ROOT:G ≈ CHORD:G_MAJOR
DYNAMIC:MF - DYNAMIC:F + DYNAMIC:P

Because atomic and compound tokenizations represent concepts differently, implement only valid analogies for the selected tokenizer.

Report top-k results, not only the first result.

18.4 Linear probes

Freeze embeddings and train small linear classifiers to predict token attributes:

* token category;
* pitch class;
* octave;
* duration class;
* hand;
* articulation;
* dynamic;
* chord quality.

Use train/validation/test splits and report accuracy or macro-F1.

Prevent leakage: do not ask a probe to predict a label directly encoded in an identical one-hot token namespace without explaining that the task is trivial.

More meaningful probe targets should include contextual attributes inferred from corpus usage, for example:

* likely hand;
* metrical position bucket;
* harmonic-function bucket, when synthetic labels are available;
* simultaneous versus sequential usage.

18.5 Retrieval evaluation

Use synthetic transposed and rhythmically varied motifs with known source families.

Measure whether the system retrieves:

* the original motif;
* transposed versions;
* rhythmic variants;
* unrelated phrases less frequently.

Report recall@k and mean reciprocal rank.

18.6 Baselines

Compare learned embeddings against:

1. random embeddings;
2. one-hot vectors, where computationally practical;
3. manually constructed musical-feature vectors;
4. untrained embedding initialization.

This comparison is mandatory. It demonstrates whether training adds useful structure.

⸻

19. Using the embeddings

Create a public API:

from pnl2vec import PNL2Vec
model = PNL2Vec.load("artifacts")
tokens = model.tokenize(pnl_text)
ids = model.encode(pnl_text)
token_vectors = model.embed_tokens(ids)
phrase_vector = model.embed_pnl(pnl_text)
neighbors = model.nearest_neighbors(
    "PITCH_CLASS:C",
    top_k=10,
    category="pitch",
)
results = model.search_similar_phrases(
    pnl_text,
    top_k=10,
)

Also support:

vector = model.embedding_for_token("DURATION:1/4")
similarity = model.similarity(
    "ARTICULATION:STACCATO",
    "ARTICULATION:ACCENT",
)

The API must return typed data structures, not raw dictionaries everywhere.

⸻

20. Optional downstream classifier demonstration

Add one small demonstration showing how trained embeddings can be reused.

Choose one task that is supported by the generated corpus:

* predict likely hand;
* classify a phrase as scale, arpeggio, chordal, or contrapuntal;
* predict harmonic-function class;
* identify motif family.

Compare:

1. frozen random embeddings;
2. frozen learned embeddings;
3. trainable learned embeddings.

Use a small mean-pooled classifier. The purpose is to demonstrate utility, not achieve state-of-the-art performance.

Save a Markdown report with the comparison.

⸻

21. Command-line interface

Implement a unified CLI:

pnl2vec validate <path>
pnl2vec generate-synthetic --size tiny
pnl2vec inspect-corpus <path>
pnl2vec build-vocab <path>
pnl2vec train --config configs/train_skipgram.yaml
pnl2vec evaluate --checkpoint artifacts/checkpoints/best.pt
pnl2vec visualize --checkpoint artifacts/checkpoints/best.pt
pnl2vec neighbors TOKEN --top-k 10
pnl2vec analogy TOKEN_A TOKEN_B TOKEN_C --top-k 10
pnl2vec embed input.pnl --output vector.npy
pnl2vec index <corpus-path> --output <index-path>
pnl2vec search query.pnl --index <index-path>

Every command must:

* provide useful --help;
* return a nonzero exit code on failure;
* print concise actionable errors;
* support explicit configuration files;
* avoid overwriting artifacts unless --force is supplied.

⸻

22. Tests

Write meaningful tests, not placeholder tests.

Required coverage includes:

* lexer behavior;
* parser behavior;
* malformed input diagnostics;
* parse/serialize round trip;
* deterministic tokenization;
* atomic tokenization;
* compound token fallback;
* vocabulary serialization;
* unknown-token behavior;
* no train/test vocabulary leakage;
* context windows not crossing documents;
* simultaneous-event pair generation;
* negative samples excluding invalid tokens;
* model output shapes;
* loss decreases on a tiny corpus;
* checkpoint save/load equivalence;
* nearest-neighbor self-exclusion;
* phrase pooling;
* deterministic synthetic generation;
* CLI smoke tests.

Create a tiny fixture corpus committed to data/examples/.

All tests must run with:

pytest

The complete test suite must not require a GPU.

⸻

23. Documentation

Write a detailed README.md containing:

1. What an embedding is.
2. How tokenization relates to embeddings.
3. Why an embedding layer is a trainable lookup table.
4. Why the vectors become meaningful only through a training objective.
5. Why tokenization design influences what the model can learn.
6. Differences between atomic and compound PNL tokenization.
7. Differences between token embeddings and contextual embeddings.
8. How skip-gram and CBOW work.
9. How musical context differs from ordinary text context.
10. How to generate a corpus.
11. How to train.
12. How to visualize.
13. How to query nearest neighbors.
14. How to perform phrase retrieval.
15. How to interpret results cautiously.
16. Known limitations.
17. How this project could evolve into a small PNL transformer.

Include a “five-minute experiment”:

pip install -e ".[dev]"
pnl2vec generate-synthetic --size tiny
pnl2vec train --config configs/train_skipgram.yaml
pnl2vec evaluate --checkpoint artifacts/checkpoints/best.pt
pnl2vec visualize --checkpoint artifacts/checkpoints/best.pt
pnl2vec neighbors "PITCH_CLASS:C" --top-k 10

⸻

24. Implementation details

Use:

* PyTorch;
* NumPy;
* PyYAML;
* Matplotlib;
* scikit-learn;
* Plotly;
* Typer or argparse;
* pytest.

Make UMAP optional:

pip install -e ".[umap]"

Use type hints throughout.

Use logging rather than scattered print statements, except for intentional CLI output.

Use dataclasses or Pydantic models for configurations and data structures.

Avoid hidden global state.

Use vectorized PyTorch operations in the training loop.

Do not store the entire set of training pairs in memory for large corpora. Implement streaming or chunked pair generation.

Show progress bars for long operations.

Use numerically stable negative-sampling loss.

Normalize embeddings only when required for similarity or visualization; preserve raw trained weights in saved artifacts.

⸻

25. Acceptance criteria

The implementation is complete only when all of the following work:

pip install -e ".[dev]"
pytest
pnl2vec generate-synthetic --size tiny
pnl2vec inspect-corpus data/raw
pnl2vec train --config configs/train_skipgram.yaml
pnl2vec evaluate --checkpoint artifacts/checkpoints/best.pt
pnl2vec visualize --checkpoint artifacts/checkpoints/best.pt
pnl2vec neighbors "PITCH_CLASS:C" --top-k 10

The training run must produce:

* a decreasing loss on the tiny synthetic corpus;
* a saved vocabulary;
* a saved checkpoint;
* exported embedding matrices;
* static visualizations;
* an interactive visualization;
* nearest-neighbor output;
* an evaluation report;
* a comparison against random and untrained baselines.

The final README must state honestly whether musically meaningful relationships emerged. Do not fabricate successful results.

⸻

26. Development sequence

Implement the project incrementally in this order:

Phase 1: inspect and plan

* Inspect the existing repository.
* Locate the canonical PNL/2 grammar and examples.
* Write IMPLEMENTATION_PLAN.md.
* List assumptions and risks.
* Do not modify the grammar without a documented reason.

Phase 2: parser and tokenizer

* Implement or integrate parsing.
* Implement token objects.
* Implement atomic tokenization.
* Implement round-trip and tokenizer tests.

Phase 3: corpus and vocabulary

* Implement corpus loading and splitting.
* Build the vocabulary.
* Generate corpus statistics.
* Implement the synthetic corpus generator.

Phase 4: training data

* Implement context-pair generation.
* Implement music-aware context policies.
* Verify that pairs never cross documents.
* Add tests and sample pair inspection.

Phase 5: embedding models

* Implement skip-gram negative sampling.
* Implement CBOW negative sampling.
* Train on a tiny corpus.
* Confirm that the loss decreases.

Phase 6: analysis

* Implement nearest neighbors.
* Implement PCA, t-SNE, and optional UMAP.
* Implement intrinsic evaluations and baselines.
* Generate reports.

Phase 7: use cases

* Implement phrase pooling.
* Implement phrase indexing and retrieval.
* Implement the downstream classifier demonstration.
* Add the public API.

Phase 8: polish

* Complete the CLI.
* Complete documentation.
* Run all tests.
* Run the five-minute experiment.
* Record actual outputs and limitations.

At the end of every phase:

1. run the relevant tests;
2. report changed files;
3. report commands executed;
4. report failures honestly;
5. fix failures before continuing when feasible.

⸻

27. Important conceptual distinctions

The implementation and documentation must preserve these distinctions:

Tokenizer

A deterministic mapping from PNL/2 source into discrete token identities.

Vocabulary

A mapping between canonical token strings and integer IDs.

Embedding layer

A trainable matrix:

vocabulary size × embedding dimension

where selecting a token ID retrieves one row.

Training objective

The mechanism that causes useful geometric relationships to emerge in the embedding matrix.

Token embedding

A static vector associated with one vocabulary item.

Event embedding

A vector created by combining the atomic tokens of a musical event.

Phrase embedding

A pooled representation of multiple token or event embeddings.

Contextual embedding

A representation whose value changes depending on surrounding events. This project does not initially train a transformer, so its basic token embeddings are not contextual.

Explain these distinctions explicitly in the README.

⸻

28. Final deliverable

Produce a functioning repository, not only an architectural description.

When complete, provide:

* a summary of the architecture;
* the exact commands needed to run it;
* the location of all generated artifacts;
* sample nearest-neighbor output;
* sample visualization filenames;
* evaluation results against baselines;
* known limitations;
* recommended next step toward a contextual PNL/2 transformer.

Begin by inspecting the repository and locating the exact PNL/2 grammar. Then create IMPLEMENTATION_PLAN.md before writing production code.