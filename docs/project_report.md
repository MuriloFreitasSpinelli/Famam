
\chapter{Goal Definition}
\label{chap:target}

This chapter defines the project goals using the MoSCoW prioritization framework. The criteria reflect what was actually implemented in the final system---including several capabilities that exceeded initial expectations.

\section{Mandatory Criteria}

\must{1}{The system must import and manage symbolic music datasets in MIDI format using \texttt{MusPy} and \texttt{Pretty MIDI}, storing them in HDF5 format for efficient access during training. A structured project directory organizes raw data, processed datasets, model checkpoints, and configuration files.}

\must{2}{The system must validate all imported MIDI files, detecting and handling corrupt files, unsupported encodings, timing inconsistencies, and missing musical information. Invalid files are logged and excluded from processing without crashing the pipeline.}

\must{3}{The system must convert MIDI files into multiple internal numerical representations suitable for sequence modeling:
  \begin{itemize}
    \item Event-based encoding (note-on, note-off, time-shift tokens)
    \item REMI encoding (bar markers, position tokens, explicit durations)
    \item Multi-track interleaved encoding (all instruments in a single sequence)
  \end{itemize}
All encodings support round-trip conversion back to playable MIDI.}

\must{4}{The system must provide configurable data preprocessing steps implemented in Python, including:
  \begin{itemize}
    \item Resolution normalization (converting to consistent ticks-per-beat)
    \item Quantization to configurable grid divisions
    \item Instrument/track filtering based on program numbers and note counts
    \item Bar-aligned segmentation with configurable segment length in bars
    \item Sequence padding and truncation to maximum sequence length
    \item Time signature detection and validation
  \end{itemize}}

\must{5}{The system must split the dataset into training, validation, and test sets at the song level (preventing segment leakage) with configurable ratios and reproducible random seeds stored in configuration files.}

\must{6}{The system must provide multiple neural network architectures for music generation implemented in \texttt{TensorFlow}/\texttt{Keras}:
  \begin{itemize}
    \item LSTM model with configurable stacked layers
    \item LSTM with self-attention mechanism for improved long-range modeling
    \item Transformer model with relative positional attention for capturing distance-based musical patterns
  \end{itemize}}

\must{7}{The system must allow comprehensive configuration of model hyperparameters via JSON configuration files (\texttt{TrainingConfig}), including:
  \begin{itemize}
    \item Model architecture selection (LSTM or Transformer)
    \item Layer counts, hidden dimensions, attention heads, feed-forward dimensions
    \item Learning rate, batch size, epochs, dropout rates
    \item Optimizer selection (Adam, AdamW, SGD, RMSprop) with configurable parameters
    \item Learning rate schedules (Transformer warmup, cosine decay, constant)
  \end{itemize}}

\must{8}{The system must train models and compute training and validation metrics (loss per epoch, perplexity), logging these via TensorBoard for real-time visualization and storing history for later analysis.}

\must{9}{The system must visualize training and validation curves through TensorBoard integration, enabling monitoring of loss, learning rate, and other metrics during long training runs.}

\must{10}{The system must save trained model checkpoints (weights, architecture, optimizer state) to disk in Keras format, with configurable checkpointing frequency. Model bundles package trained models with their encoders and vocabularies for portable deployment.}

\must{11}{The system must provide generation functions that use trained models to generate new musical sequences with configurable sampling strategies:
  \begin{itemize}
    \item Temperature scaling for controlling randomness
    \item Top-k filtering to restrict sampling to most likely tokens
    \item Nucleus (top-p) sampling for dynamic candidate pool sizing
    \item Minimum length enforcement to prevent premature termination
  \end{itemize}
Generated output converts back to playable MIDI format via \texttt{MusPy}.}

\must{12}{The system must provide interactive command-line interfaces (CLIs) for end-to-end workflows:
  \begin{itemize}
    \item Experiment CLI for dataset creation, configuration management, and model training
    \item Generation CLI for loading trained models and generating music with interactive parameter adjustment
  \end{itemize}
The system is also usable programmatically through Python APIs.}

\must{13}{The system must ensure reproducibility by storing all configuration parameters (random seeds, preprocessing options, model hyperparameters, encoder settings) in serializable JSON configuration files that fully specify each experiment.}

\must{14}{The system must provide clear summaries of model architecture, parameter counts, and training configuration through model \texttt{summary()} methods and configuration display utilities.}

\must{15}{The system must support distributed training across multiple GPUs using TensorFlow's \texttt{MirroredStrategy} for single-node multi-GPU training.}

\must{16}{The system must allow configuration of distributed training parameters (distribution strategy, batch size scaling) via the central \texttt{TrainingConfig}.}

\must{17}{The system must support polyphonic multi-track music generation where all instruments are generated together with awareness of inter-instrument relationships, not just monophonic single-track generation.}

\must{18}{The system must support genre conditioning, allowing users to specify desired musical genre during generation through conditioning tokens.}

\must{19}{The system must support instrument conditioning, allowing users to specify which instruments to include in generated output.}

\section{Preferred Criteria}

\should{1}{The system should provide automatic dataset augmentation methods to increase training diversity. \textbf{Implemented}: transposition (configurable semitone range) and tempo variation (configurable BPM range and steps).}

\should{2}{The system should include mechanisms to detect and mitigate overfitting. \textbf{Implemented}: validation loss monitoring, early stopping with configurable patience and minimum delta, and dropout regularization.}

\should{3}{The system should allow exporting training logs and metrics in standard formats. \textbf{Implemented}: TensorBoard logs, JSON configuration export, Keras model format.}

\should{4}{The system should include configuration presets for different training scenarios. \textbf{Implemented}: \texttt{get\_transformer\_small()}, \texttt{get\_transformer\_medium()}, \texttt{get\_transformer\_large()}, \texttt{get\_lstm\_small()}, \texttt{get\_lstm\_medium()} preset functions in \texttt{TrainingConfig}.}

\should{5}{The system should support gradient clipping to stabilize training. \textbf{Implemented}: configurable gradient clipping with adjustable clip value.}

\should{6}{The system should implement label smoothing for improved generalization. \textbf{Implemented}: configurable label smoothing in loss computation.}

\should{7}{The system should provide masked loss computation that ignores padding tokens. \textbf{Implemented}: \texttt{MaskedSparseCategoricalCrossentropy} loss function.}

\should{8}{The system should support continuation generation from existing music prompts. \textbf{Implemented}: \texttt{generate\_with\_prompt()} method in \texttt{PolyGenerator}.}

\should{9}{The system should support extended generation through segment concatenation. \textbf{Implemented}: \texttt{generate\_extended()} method for producing longer compositions.}

\section{Optional Criteria}

\could{1}{The system may support mixed-precision training to improve efficiency on compatible hardware. \textbf{Implemented}: float16 support where numerical stability permits.}

\could{2}{The system may implement gradient checkpointing to reduce memory usage during training. \textbf{Implemented}: activation recomputation during backpropagation for memory-constrained scenarios.}

\could{3}{The system may provide GPU memory management to prevent out-of-memory errors. \textbf{Implemented}: TensorFlow memory growth configuration to avoid greedy GPU memory allocation.}

\could{4}{The system may support checkpoint-based job resumption for long training runs. \textbf{Implemented}: training automatically resumes from latest checkpoint when restarted.}

\could{5}{The system may generate an automated experiment report with dataset statistics and configuration summary. \textbf{Partially implemented}: configuration serialization and model summaries provide most reporting functionality.}

\section{Exclusion Criteria}

\wont{1}{The system will not provide real-time audio synthesis or advanced sound design. Only symbolic music generation (MIDI output) is within scope; rendering to audio requires external software.}

\wont{2}{The system will not include automatic detection, classification, or transcription of audio files. Only symbolic MIDI input is supported; raw audio processing is out of scope.}

\wont{3}{The system will not include a recommendation engine or playlist generation. The focus is on music creation, not music discovery.}

\wont{4}{The system will not include multi-user account management or authentication mechanisms. This is a research tool, not a multi-tenant service.}

\wont{5}{The system will not provide a graphical user interface (GUI) or web-based interface. Interactions occur through command-line interfaces and programmatic Python APIs.}

\wont{6}{The system will not generate long-form structured compositions with distinct movements or sections (verse-chorus-bridge structure). Generation produces segments that can be concatenated but does not understand high-level song structure.}

\wont{7}{The system will not guarantee professional musical quality or artistic creativity. Output quality depends on training data quality and model capacity; results are suitable for research and experimentation but not professional music production without human curation.}

\wont{8}{The system will not support real-time interactive generation or live performance applications. Generation is a batch process that produces complete sequences.}



\chapter{Non-Functional Requirements}
\label{chap:non_functional_req}

Beyond what the system does, these requirements specify how well it does it---the quality attributes that determine whether the system is actually pleasant to use, robust under real-world conditions, and maintainable over time.

\section{Usability}

\begin{itemize}

\item \qualityReq{10}{The system must provide interactive command-line interfaces with menu-driven navigation, enabling users to perform complex workflows (dataset creation, training, generation) without memorizing command syntax or reading source code. \textbf{Implemented}: Experiment CLI and Generation CLI with numbered menus, input validation, and clear prompts.}

\item \qualityReq{20}{Configuration must be manageable through human-readable JSON files, not hardcoded values or obscure command-line flags. Users should be able to inspect and modify configurations with any text editor. \textbf{Implemented}: \texttt{MusicDatasetConfig} and \texttt{TrainingConfig} serialize to readable JSON with descriptive field names.}

\item \qualityReq{30}{The system must provide clear feedback during long-running operations. Users should never stare at a frozen terminal wondering if something is happening. \textbf{Implemented}: Progress indicators during preprocessing, TensorBoard integration for training monitoring, epoch-by-epoch loss reporting.}

\item \qualityReq{40}{Error messages must be informative and actionable, telling users what went wrong and suggesting how to fix it---not just stack traces. \textbf{Implemented}: Validation errors in configuration loading, descriptive exceptions in preprocessing pipeline.}

\end{itemize}

\section{Reliability and Fault Tolerance}

\begin{itemize}

\item \qualityReq{50}{The system must handle corrupt, malformed, or invalid MIDI files gracefully, logging problems and continuing with valid files rather than crashing the entire preprocessing run. \textbf{Implemented}: Try-catch wrappers around file loading with problematic file logging.}

\item \qualityReq{60}{Training must auto-resume from checkpoints after crashes, power failures, or job scheduler terminations. Hours of training progress must never be lost to infrastructure issues. \textbf{Implemented}: Keras checkpointing with configurable frequency, automatic latest-checkpoint detection on restart.}

\item \qualityReq{70}{The system must validate configurations at load time, catching invalid parameter combinations before expensive operations begin. \textbf{Implemented}: \texttt{\_\_post\_init\_\_} validation in dataclasses, early failure on nonsensical settings.}

\item \qualityReq{80}{GPU memory management must prevent out-of-memory crashes by using TensorFlow's memory growth configuration rather than greedy allocation. \textbf{Implemented}: Memory growth enabled by default in training pipeline.}

\item \qualityReq{90}{The system must handle missing genre metadata gracefully, either excluding files without annotations or using default values, rather than failing on incomplete datasets. \textbf{Implemented}: Configurable genre filtering with support for untagged files.}

\end{itemize}

\section{Performance}

\begin{itemize}

\item \qualityReq{100}{Music generation must complete in reasonable time for interactive use. A 4-bar multi-track segment should generate in under 10 seconds on CPU, under 2 seconds with GPU acceleration. \textbf{Achieved}: Autoregressive generation with efficient token sampling meets these targets on typical hardware.}

\item \qualityReq{110}{MIDI export must be near-instantaneous. Converting generated tokens back to a playable MIDI file should complete in under 1 second regardless of sequence length. \textbf{Achieved}: MusPy export is effectively instant for typical generation lengths.}

\item \qualityReq{120}{Model loading must be fast enough for practical use. Loading a trained model bundle (including encoder and vocabulary) should complete in under 5 seconds. \textbf{Achieved}: Keras model loading with bundled metadata typically completes in 2-4 seconds.}

\item \qualityReq{130}{Dataset preprocessing must scale reasonably with corpus size. Processing the full 17,000-song Lakh MIDI Clean dataset should complete in under 2 hours on typical hardware. \textbf{Achieved}: Parallel-friendly preprocessing pipeline meets this target.}

\item \qualityReq{140}{Training throughput must efficiently utilize available GPU resources. Batch processing and data loading should not bottleneck GPU computation. \textbf{Implemented}: HDF5 efficient access patterns, prefetching, sequence length bucketing.}

\end{itemize}

\section{Scalability}

\begin{itemize}

\item \qualityReq{150}{The system must support datasets ranging from dozens to tens of thousands of MIDI files without code modification---only configuration changes. \textbf{Implemented}: Configuration-driven preprocessing handles any corpus size.}

\item \qualityReq{160}{Training must support distributed execution across multiple GPUs using TensorFlow's distribution strategies, with graceful fallback to single-GPU or CPU-only operation. \textbf{Implemented}: \texttt{MirroredStrategy} support with automatic device detection.}

\item \qualityReq{170}{Sequence length limits must be configurable to balance memory usage against musical context. The system should handle sequences from 256 to 2048+ tokens depending on available resources. \textbf{Implemented}: \texttt{max\_seq\_length} configuration with dynamic padding.}

\item \qualityReq{180}{The system must handle varying model sizes from small debugging configurations to large production models without architectural changes. \textbf{Implemented}: Configuration presets from \texttt{get\_transformer\_small()} to \texttt{get\_transformer\_large()}.}

\end{itemize}

\section{Reproducibility}

\begin{itemize}

\item \qualityReq{190}{Experiments must be fully reproducible given the same configuration file. All parameters affecting results---random seeds, preprocessing options, model hyperparameters, training settings---must be captured in serializable configurations. \textbf{Implemented}: Complete configuration serialization to JSON.}

\item \qualityReq{200}{Random seeds must be settable at all levels (Python, NumPy, TensorFlow) to enable deterministic execution where hardware permits. \textbf{Implemented}: Seed configuration in both dataset and training configs.}

\item \qualityReq{210}{Dataset splits must be reproducible. Given the same configuration and random seed, train/validation/test splits must produce identical results. \textbf{Implemented}: Seeded splitting at song level.}

\item \qualityReq{220}{Model bundles must be self-contained, packaging everything needed to reproduce generation (model weights, encoder state, vocabulary) in a single portable artifact. \textbf{Implemented}: \texttt{ModelBundle} class with complete serialization.}

\end{itemize}

\section{Portability}

\begin{itemize}

\item \qualityReq{230}{The system must run on Windows, macOS, and Linux without platform-specific code paths or manual modifications. \textbf{Implemented}: Pure Python with cross-platform dependencies, tested on all three platforms.}

\item \qualityReq{240}{The system must work in both local development environments (workstations, laptops) and HPC cluster environments (SLURM, PBS schedulers) with only configuration changes. \textbf{Implemented}: Configurable paths, checkpoint-based job chaining support.}

\item \qualityReq{250}{Model bundles trained on one platform must be usable on any other supported platform. A model trained on a Linux cluster must generate music on a Windows laptop. \textbf{Implemented}: Platform-independent serialization formats.}

\item \qualityReq{260}{The system must function with or without GPU acceleration, automatically detecting available hardware and adapting accordingly. \textbf{Implemented}: TensorFlow automatic device placement with CPU fallback.}

\end{itemize}

\section{Maintainability}

\begin{itemize}

\item \qualityReq{270}{The codebase must follow a modular architecture with clear separation of concerns. Data preprocessing, model definition, training orchestration, and generation must be independent modules that can be modified without cascading changes. \textbf{Implemented}: Separate packages for \texttt{data}, \texttt{models}, \texttt{training}, \texttt{generation}, \texttt{cli}.}

\item \qualityReq{280}{New encoding schemes must be addable by implementing a single abstract interface (\texttt{BaseEncoder}) without modifying training or generation code. \textbf{Implemented}: Three encoders (Event, REMI, Multi-track) share common interface.}

\item \qualityReq{290}{New model architectures must be addable by implementing a single abstract interface (\texttt{BaseMusicModel}) without modifying data preprocessing or generation code. \textbf{Implemented}: LSTM and Transformer share common interface.}

\item \qualityReq{300}{Configuration classes must use Python dataclasses with type hints, enabling IDE support, automatic validation, and self-documenting code. \textbf{Implemented}: Typed dataclasses throughout configuration system.}

\end{itemize}

\section{Musical Validity}

\begin{itemize}

\item \qualityReq{310}{Generated output must contain only valid MIDI events: note pitches in range 0--127, velocities in range 1--127, non-negative timestamps. Invalid events must never appear in exported files. \textbf{Implemented}: Encoder decode methods enforce MIDI validity constraints.}

\item \qualityReq{320}{Generated music must respect basic musical structure: notes must have positive durations, drum events must use appropriate MIDI channel conventions, instrument programs must be valid General MIDI values. \textbf{Implemented}: Encoding schemes enforce structural validity.}

\item \qualityReq{330}{Bar and position tokens in REMI and Multi-track encodings must map to correct temporal positions in exported MIDI, preserving the metrical structure learned during training. \textbf{Implemented}: Round-trip encode-decode validation confirms temporal accuracy.}

\end{itemize}

\section{Documentation and Internationalization}

\begin{itemize}

\item \qualityReq{340}{All user-facing text, CLI prompts, error messages, and documentation must be in English. \textbf{Implemented}: English throughout.}

\item \qualityReq{350}{Code must include docstrings for public classes and methods, enabling IDE tooltips and automated documentation generation. \textbf{Implemented}: Comprehensive docstrings in public APIs.}

\item \qualityReq{360}{Configuration files must use descriptive field names that explain their purpose without requiring documentation lookup. \textbf{Implemented}: Self-documenting field names like \texttt{segment\_bars}, \texttt{enable\_transposition}, \texttt{max\_seq\_length}.}

\end{itemize}



\chapter{User Interface}
\label{chap:user_interface}

The system provides two interactive command-line interfaces (CLIs) that serve as the primary means of user interaction. These interfaces prioritize usability through menu-driven navigation, sensible defaults, and clear feedback---making complex machine learning workflows accessible without requiring users to memorize command syntax or dive into source code.

\section{Design Philosophy}

\subsection{Why Command-Line Interfaces?}

The decision to build CLIs rather than graphical interfaces was deliberate. Command-line interfaces offer several advantages for research tools:

\begin{itemize}
    \item \textbf{Universal Compatibility}: CLIs work identically across Windows, macOS, and Linux without platform-specific GUI frameworks. They function equally well over SSH connections to remote servers and HPC clusters.
    \item \textbf{Scriptability}: While the interactive menus serve casual use, the underlying functions remain callable from scripts for batch processing and automation.
    \item \textbf{Low Overhead}: No GUI framework dependencies, no window managers required, no graphical rendering overhead. The interface works in any terminal.
    \item \textbf{Focus on Functionality}: Development effort goes into the core system rather than interface aesthetics. A well-designed CLI can be more efficient than a mediocre GUI.
\end{itemize}

\subsection{Menu-Driven Navigation}

Both CLIs use a consistent menu-driven navigation pattern. Users never need to memorize commands or flags---options are always visible, numbered for quick selection, and organized into logical hierarchies. The pattern follows a simple structure:

\begin{verbatim}
============================================================
  Menu Title
============================================================
  [1] First Option
  [2] Second Option
  [3] Third Option
------------------------------------------------------------
  [0] Back / Exit
============================================================

Select option: _
\end{verbatim}

This structure appears throughout the system, providing consistency and predictability. Users quickly learn that [0] always returns to the previous level, numbers select options, and headers indicate context.

\subsection{Input Handling}

User input follows consistent patterns across both CLIs:

\begin{itemize}
    \item \textbf{Defaults}: Most prompts show a default value in brackets. Pressing Enter accepts the default, reducing keystrokes for common cases.
    \item \textbf{Validation}: Numeric inputs validate against allowed ranges, with clear error messages when values fall outside bounds.
    \item \textbf{Boolean Prompts}: Yes/no questions accept multiple forms (y/yes/true/1 for affirmative, anything else for negative).
    \item \textbf{List Selection}: When selecting from lists (files, configurations), users can enter either the item number or the full name.
    \item \textbf{Graceful Interrupts}: Ctrl+C and EOF conditions are caught and handled gracefully rather than crashing with stack traces.
\end{itemize}


\section{Experiment CLI}

The Experiment CLI manages the research workflow: creating configurations, building datasets, training models, and packaging results. It is the interface researchers use during development and experimentation.

\subsection{Main Menu Structure}

\begin{verbatim}
============================================================
  Experiment CLI - Dataset & Training
============================================================
  [1] Config Management
  [2] Dataset Operations
  [3] Model Training
  [4] View Information
------------------------------------------------------------
  [0] Back / Exit
============================================================
\end{verbatim}

\subsection{Config Management}

The configuration management submenu handles creation and inspection of the JSON configuration files that govern all experimental parameters.

\begin{verbatim}
============================================================
  Config Management
============================================================
  [1] Create Dataset Config
  [2] Create Training Config
  [3] List Configs
  [4] View Config
------------------------------------------------------------
  [0] Back / Exit
============================================================
\end{verbatim}

\textbf{Dataset Configuration Wizard.} Creating a dataset configuration walks the user through all preprocessing parameters in organized sections:

\begin{enumerate}
    \item \textbf{Basic Info}: Configuration name, input directories, output path
    \item \textbf{Genre Filtering}: Genre TSV file path, allowed genres list
    \item \textbf{Track Filtering}: Minimum/maximum tracks, minimum notes per track
    \item \textbf{Duration Filtering}: Minimum and maximum song duration
    \item \textbf{Resolution \& Encoding}: Ticks per quarter note, encoder type (multitrack/event/remi), maximum sequence length, velocity encoding
    \item \textbf{Segmentation}: Enable/disable segmentation, segment length in ticks
    \item \textbf{Augmentation}: Transposition and tempo variation settings
    \item \textbf{Processing}: Maximum samples, random seed
\end{enumerate}

Each section groups related parameters, with sensible defaults that work for most use cases. The wizard produces a complete \texttt{MusicDatasetConfig} JSON file ready for dataset building.

\textbf{Training Configuration Wizard.} The training configuration wizard similarly guides users through model and training parameters:

\begin{enumerate}
    \item \textbf{Model Identification}: Model name, model type (transformer/lstm)
    \item \textbf{Architecture}: Embedding dimension, dropout rate, architecture-specific parameters (layers, heads, feed-forward dimension for Transformers; units, bidirectional for LSTMs)
    \item \textbf{Training Hyperparameters}: Batch size, epochs, learning rate, warmup steps, label smoothing
    \item \textbf{Optimizer}: Optimizer selection (adam/adamw/sgd), weight decay
    \item \textbf{Learning Rate Schedule}: Schedule type (transformer/cosine/constant)
    \item \textbf{Regularization}: Gradient clipping settings
    \item \textbf{Early Stopping}: Enable/disable with patience configuration
    \item \textbf{Checkpointing}: Checkpoint saving, best-only option, TensorBoard logging
    \item \textbf{Output}: Output directory, random seed
\end{enumerate}

\subsection{Dataset Operations}

\begin{verbatim}
============================================================
  Dataset Operations
============================================================
  [1] Build Dataset from Config
  [2] View Dataset Info
  [3] View Instrument Statistics
------------------------------------------------------------
  [0] Back / Exit
============================================================
\end{verbatim}

\textbf{Build Dataset.} Lists available dataset configurations, loads the selected config, displays a summary of what will be built, and asks for confirmation before starting the potentially long preprocessing operation. Progress feedback shows files processed and any errors encountered.

\textbf{View Dataset Info.} Loads a completed dataset and displays comprehensive statistics: entry count, track count, total notes, resolution, sequence length, genre distribution, and active instruments.

\textbf{Instrument Statistics.} Shows detailed instrument usage across the dataset, helping users understand the distribution of instruments in their training data.

\subsection{Model Training}

\begin{verbatim}
============================================================
  Model Training
============================================================
  [1] Train Model from Config
  [2] Create Bundle from Checkpoint
------------------------------------------------------------
  [0] Back / Exit
============================================================
\end{verbatim}

\textbf{Train Model.} The training workflow:
\begin{enumerate}
    \item Prompts for dataset file path (lists available datasets)
    \item Prompts for training config file (lists available configs)
    \item Loads and displays both configurations for review
    \item Asks for validation split ratio
    \item Confirms before starting training
    \item Displays progress via TensorBoard integration
    \item Saves model bundle upon completion
\end{enumerate}

\textbf{Create Bundle.} Packages a trained checkpoint into a portable model bundle. Prompts for checkpoint file, training config, and dataset (for vocabulary), then creates a self-contained bundle that can be loaded by the Generation CLI.


\section{Generation CLI}

The Generation CLI is the user-facing interface for creating music from trained models. It focuses on simplicity and immediate usability---load a model, adjust a few parameters, generate music.

\subsection{Main Menu Structure}

\begin{verbatim}
============================================================
  Main Menu
============================================================
  [1] Load Model
  [2] View Model Info
  [3] View Genres
  [4] View Instruments
  [5] Generate Music
  [6] Generate Extended (Concatenation)
  [7] Generation Settings
------------------------------------------------------------
  [0] Back / Exit
============================================================
  Model: rock_transformer_v2
  Genres: 12
  Instruments: 45
\end{verbatim}

The main menu always shows the currently loaded model's status, so users know what they are working with.

\subsection{Model Loading}

The Load Model option searches for model bundles in the \texttt{models/} directory, displays found bundles with numbered selection, and loads the chosen bundle. Upon successful loading, displays:

\begin{itemize}
    \item Model name and type (Transformer/LSTM)
    \item Vocabulary size
    \item Maximum sequence length
    \item Available genres (if vocabulary included)
    \item Active instruments count
\end{itemize}

\subsection{Vocabulary Exploration}

\textbf{View Genres.} Lists all genres in the model's vocabulary with their IDs, enabling users to select appropriate genre conditioning for generation.

\textbf{View Instruments.} Shows instrument usage statistics from training, helping users understand what instruments the model learned and how frequently they appeared.

\subsection{Music Generation}

The core generation workflow:

\begin{enumerate}
    \item Displays current model info and available genres
    \item Prompts for genre ID (from the displayed list)
    \item Shows current generation settings (temperature, top-k, top-p)
    \item Optionally allows modifying settings
    \item Prompts for maximum sequence length
    \item Prompts for minimum length before allowing EOS
    \item Asks whether to ignore EOS (generate full length)
    \item Asks whether to exclude drums from output
    \item Prompts for output file path
    \item Confirms and generates
    \item Displays results: token count, track breakdown, note counts per track
\end{enumerate}

\subsection{Extended Generation}

For longer compositions, the Extended Generation option concatenates multiple generation segments:

\begin{enumerate}
    \item Prompts for genre ID
    \item Offers two length specification methods:
    \begin{itemize}
        \item By number of segments to concatenate
        \item By approximate duration in bars (automatically calculates segments needed)
    \end{itemize}
    \item Prompts for tokens per segment
    \item Shows and optionally modifies generation settings
    \item Generates each segment with progress feedback
    \item Combines all segments with proper time offsets
    \item Saves the concatenated result
    \item Displays total duration, track count, and note statistics
\end{enumerate}

\subsection{Generation Settings}

A dedicated settings menu allows adjusting generation parameters without going through the full generation flow:

\begin{verbatim}
============================================================
  Generation Settings
============================================================

  Current settings:
    [1] Temperature: 0.9
    [2] Top-k: 50
    [3] Top-p: 0.92
    [4] Min length: 256
------------------------------------------------------------
    [0] Back
============================================================
\end{verbatim}

Settings persist across generation sessions, so users can tune parameters once and generate multiple pieces with the same configuration.

\subsection{Generation Parameters Explained}

The CLI provides access to all sampling parameters:

\begin{itemize}
    \item \textbf{Temperature} (0.1--2.0): Controls randomness. Lower values produce more conservative, predictable output; higher values increase variety but risk incoherence. Default 0.9 balances creativity with stability.

    \item \textbf{Top-k} (0--500): Restricts sampling to the k most likely tokens. 0 disables the filter. Default 50 prevents extremely unlikely tokens while maintaining variety.

    \item \textbf{Top-p / Nucleus Sampling} (0.0--1.0): Dynamically selects tokens until cumulative probability reaches p. Default 0.92 adapts the candidate pool based on the model's confidence.

    \item \textbf{Min Length}: Prevents the model from generating EOS (end of sequence) until this many tokens have been produced. Ensures minimum musical content.
\end{itemize}


\section{Common UI Patterns}

Both CLIs share common helper functions that ensure consistent behavior:

\begin{itemize}
    \item \texttt{clear\_screen()}: Clears terminal for fresh menu display
    \item \texttt{print\_header(title)}: Displays styled section headers
    \item \texttt{print\_menu(title, options)}: Renders numbered menu with back/exit option
    \item \texttt{get\_input(prompt, default)}: Prompts with optional default value
    \item \texttt{get\_int(prompt, default, min, max)}: Integer input with range validation
    \item \texttt{get\_float(prompt, default, min, max)}: Float input with range validation
    \item \texttt{get\_bool(prompt, default)}: Yes/no prompt
    \item \texttt{get\_choice(prompt, max)}: Menu selection
    \item \texttt{wait\_for\_enter()}: Pauses for user acknowledgment
\end{itemize}

This shared infrastructure ensures that users who learn one CLI can immediately use the other with no learning curve.


\chapter{Design}

\section{The Dataset}

The Lakh MIDI dataset exists in multiple versions, with the full Lakh MIDI dataset representing the most comprehensive collection. However, this comprehensiveness comes at a cost---the dataset was scraped from various web sources and contains files requiring extensive cleaning and preprocessing. To avoid unnecessary data curation effort while maintaining sufficient scale for model training, we adopted the Lakh MIDI Clean dataset, a curated subset that has undergone systematic quality filtering and provides adequate musical diversity for our purposes.

\subsection{Clean MIDI Dataset}

The Clean MIDI dataset comprises approximately 17,000 songs spanning multiple genres and musical styles. This scale proves more than sufficient for training purposes, particularly when combined with segmentation and augmentation strategies. Through bar-aligned segmentation, each song yields multiple training samples, and augmentation techniques such as transposition and tempo variation further expand the effective dataset size. Conservative estimates suggest the preprocessing pipeline generates approximately 500,000 training samples from the base corpus---substantially exceeding the data requirements for convergent model training even with large-capacity architectures.

\subsection{Genre Mappings}

Genre metadata for the Clean MIDI dataset was obtained from MIDI Explorer \newline (\texttt{https://midiexplorer.sourceforge.io/}), a tool specifically designed for analyzing and annotating this dataset. The pre-computed genre mappings enable genre-conditioned music generation, allowing the model to learn style-specific patterns and generate music consistent with particular genre conventions. This genre awareness supports training a unified model with context spanning multiple musical styles, enabling controlled generation based on desired genre characteristics. Utilizing the complete genre-annotated dataset with an appropriately scaled architecture ensures that data availability will not constrain model performance or generalization capabilities.

\section{Design}

\subsection{Prototype}

Prior to formal design, a prototype implementation was developed to establish foundational understanding of the problem domain. This exploratory phase was essential given the unfamiliarity with both Python conventions and the unique challenges of machine learning research workflows.

The prototype served multiple purposes: identifying effective encoding strategies, establishing data pipeline patterns, understanding library conventions (MusPy, TensorFlow, Keras), and determining which components required flexibility for experimentation. Key questions addressed included monophonic versus polyphonic encoding trade-offs, sequence length implications, and the practical challenges of training large sequence models.

Through this iterative process, three critical areas emerged as requiring maximum flexibility: encoding schemes, model architectures, and generation strategies. The sequence-to-sequence nature of all approaches meant that supporting both monophonic and polyphonic generation primarily required different encoder and generator implementations rather than fundamental architectural changes.

The prototype revealed several important lessons that informed the final design: bar-aligned segmentation significantly improves musical coherence compared to arbitrary tick-based segmentation; vocabulary management requires careful attention to ensure genre and instrument information propagates correctly through the pipeline; and multi-track encoding enables learning of inter-instrument relationships that single-track approaches cannot capture.

\subsection{Note on Music Theory}

\textbf{Temporal Structure and Bar Segmentation.} Musical organization fundamentally relies on hierarchical temporal structure, with measures (bars) serving as the primary unit of musical phrase construction. Segmenting training data on bar boundaries rather than arbitrary time divisions is critical for several reasons. First, musical patterns naturally align with bar structure---melodies, chord progressions, and rhythmic motifs are conceived and executed in multiples of bars (typically 4, 8, or 16). Second, bar-aligned segmentation ensures that the model learns complete musical ideas rather than fragmented patterns that begin or end mid-phrase. Training with 4/4 time signature segmentation (4 beats per bar) or even finer 16th-note resolution allows the model to internalize the metrical grid that underpins Western music, significantly improving pattern recognition and generation accuracy. While this segmentation increases training complexity due to the need for precise alignment and metadata extraction, the resulting improvement in musical coherence and structural understanding justifies the additional computational overhead.

\textbf{The Role of Drums in Rhythmic Foundation.} Percussion instruments, particularly drums, serve as the rhythmic backbone of most musical compositions. The drum track establishes the tempo, emphasizes the metrical structure, and provides rhythmic anchor points that guide both performers and listeners. In multi-track music generation, the drum sequence often determines whether a piece feels cohesive or disjointed. The kick drum typically aligns with strong beats (beats 1 and 3 in 4/4 time), while the snare accents weak beats (2 and 4), creating the fundamental pulse that other instruments reference. Hi-hats and cymbals provide subdivision and forward momentum. A generative model that fails to maintain consistent, metrically-aligned drum patterns will produce music that sounds amateurish or incoherent, regardless of melodic or harmonic quality. Consequently, encoding drum events with high temporal precision and ensuring the model learns their structural role is paramount for realistic music generation.

\textbf{Inter-Instrument Relationships.} Music is inherently polyphonic, with instruments occupying complementary roles within a composition. The bass guitar typically provides harmonic foundation and rhythmic support, often locking with the kick drum to create a unified low-frequency pulse. Harmonic instruments (guitar, piano, synthesizers) fill the middle register with chord progressions and textural elements. Melodic instruments or vocals occupy the upper register, carrying the primary musical narrative. These instruments do not operate independently---their interactions create the musical texture. For instance, a bass note typically changes on strong beats (1 and 3) to outline chord changes, while melodic phrases often begin on or anticipate these changes. A successful generative model must learn these inter-instrument dependencies, understanding that a chord change in the piano part should trigger corresponding adjustments in the bass line and that melodic phrasing should respect the underlying harmonic rhythm.

\textbf{Human Timing Perception and Microtiming.} Human musical perception exhibits fascinating tolerance for---and even preference toward---imperfect timing. Perfectly quantized music, where every note aligns exactly to the grid, often sounds mechanical and lifeless. Professional musicians introduce subtle timing deviations called microtiming: notes played slightly ahead of or behind the beat, note durations that deviate from exact values, and small variations in tempo (rubato). These imperfections create groove, swing, and emotional expression. For example, swing rhythm in jazz deliberately delays the second eighth note of each beat pair, creating a ``long-short'' feel that cannot be precisely notated. Similarly, human drummers naturally introduce slight timing variations that make the rhythm feel organic. However, these deviations must remain within perceptual thresholds---deviations beyond approximately 20--50 milliseconds begin to sound like errors rather than expressive choices. Generative models trained exclusively on quantized MIDI data may produce music that lacks this human quality. Some approaches address this by training on performance MIDI with preserved timing nuances, while others introduce controlled randomness during generation. The challenge lies in teaching the model to distinguish between musically meaningful imperfections and actual mistakes.

\textbf{Harmonic and Melodic Constraints.} Not all note combinations are perceptually equal. Western music theory has identified intervals and chord structures that sound consonant (pleasant, stable) versus dissonant (tense, unstable). While dissonance has artistic value when used intentionally and resolved properly, random dissonance sounds objectionable to most listeners. Common problematic patterns include:
\begin{itemize}
    \item Minor second intervals (adjacent notes) without harmonic justification
    \item Unresolved tritones (augmented fourth/diminished fifth intervals)
    \item Melodies that leap by large intervals (e.g., major seventh) without stepwise approach or resolution
    \item Chord progressions that violate voice leading principles (e.g., parallel fifths in classical contexts)
    \item Notes outside the prevailing key signature without chromatic preparation
\end{itemize}
A model trained on high-quality musical data implicitly learns these constraints through statistical patterns---certain note combinations simply occur more frequently than others. However, pure statistical learning may not fully capture the theoretical rules that govern harmonic function. Genre-specific constraints add further complexity: jazz tolerates extended and altered chords that would sound wrong in pop music, while atonal contemporary classical music intentionally violates traditional harmonic rules. The model must learn not just what sequences are statistically common, but what makes them musically coherent within a given stylistic context.

\textbf{Implications for Model Design.} These music-theoretic considerations directly inform encoding and architecture choices. Bar-aligned segmentation with explicit position tokens (as in REMI encoding) helps the model learn metrical structure. Multi-track encoding with instrument-specific tokens allows learning of inter-instrument relationships. Relative positional attention mechanisms enable the model to recognize that ``every 4 beats'' and ``every 16 beats'' represent musically significant intervals regardless of absolute position. Training on diverse, high-quality datasets ensures exposure to proper voice leading, harmonic progressions, and stylistic conventions. While these approaches increase model complexity and training requirements, they are essential for generating music that satisfies both statistical patterns and perceptual expectations---music that sounds not just plausible, but genuinely musical.

\subsection{Design Philosophy}

\textbf{Research Reproducibility.} The system prioritizes reproducibility through comprehensive experiment tracking and configuration management. All experimental parameters, from data preprocessing settings to model hyperparameters, are serialized to enable exact replication of results. This approach ensures that successful configurations can be reliably reused and that comparative experiments maintain consistency across runs. The design emphasizes transparency in the complete pipeline, documenting encodings, transformations, and training conditions to facilitate scientific validation and iterative improvement.

\textbf{Abstractions for Experimentation and Extensions.} The architecture employs systematic abstraction layers to facilitate experimentation and future extensions. By identifying common patterns and creating well-defined interfaces during the design phase, the system enables researchers to modify individual components without affecting the broader pipeline. This modularity allows for rapid prototyping of new encoding schemes, model architectures, or training strategies. Even when only a single implementation currently exists, the abstraction framework prepares the codebase for alternative approaches, reducing the barrier to future research contributions.

\textbf{Flexibility.} The system supports both single-track and multi-track music generation, recognizing that different musical contexts require different representations. Single-track generation focuses on melodic or instrumental solos, while multi-track generation captures the interplay between multiple instruments performing simultaneously. This flexibility extends to supporting multiple encoding schemes, each optimized for different use cases, and allowing researchers to select the most appropriate representation for their specific generation task without architectural constraints.

\textbf{Musical Domain Awareness.} Music generation cannot succeed without grounding in music theory and the fundamental principles that govern musical structure. Simply treating music as arbitrary sequences ignores the essential characteristics that make music coherent and aesthetically pleasing. The system incorporates domain knowledge at multiple levels: recognizing that drums drive rhythmic structure, that music organizes around bar and positional hierarchies, that genre conventions shape compositional choices, and that instrument roles define textural relationships. By encoding these musical fundamentals---such as metrical position, instrument identity, and relative temporal distances---the system provides models with the structural scaffolding necessary to learn genuine musical patterns rather than superficial statistical correlations. This domain-aware design philosophy acknowledges that teaching a model to generate music requires translating the essence of musical structure into representations the model can process and internalize.

\textbf{Pipeline Separation.} The generation pipeline follows a modular, stage-based architecture where each phase operates independently with clearly defined inputs and outputs. Data preprocessing, model training, and music generation constitute separate stages that can be executed individually or as an integrated workflow. This separation enables researchers to resume work from any pipeline stage given the appropriate intermediate data, facilitating debugging, experimentation with individual components, and parallelization of different stages across computing resources. Configuration files govern each stage, ensuring that the complete pipeline can execute from start to finish without manual intervention once properly configured.

\textbf{Augmentation as First-Class Consideration.} Data availability and quality fundamentally determine model performance, particularly for large-scale generative models. The system treats data augmentation not as an afterthought but as an integral component of the preprocessing pipeline. Augmentation techniques such as transposition, tempo modification, and time stretching artificially expand the training corpus, exposing models to greater musical variety. Additionally, the segmentation strategy generates multiple training samples from each song while maintaining metadata links between segments through song identifiers. This approach preserves musical coherence---allowing the system to track which segments originated from the same composition---while maximizing data utilization. The result is substantially larger effective training sets without requiring proportionally more source material.

\textbf{Practical Deployment.} The system acknowledges the computational demands of training large sequence models and incorporates features necessary for practical deployment in high-performance computing environments. GPU resource management ensures efficient utilization of available hardware, while distributed training support enables scaling across multiple nodes in cluster environments such as BwUniCluster 3.0. Training incorporates checkpoint mechanisms to preserve progress and enable recovery from interruptions, along with early stopping criteria to prevent overfitting and optimize computational resource usage. The system employs portable model bundle formats that facilitate transfer between cluster training environments and local machines for generation and evaluation, streamlining the research workflow across different computing contexts.


\subsection{Software Architecture}

\textbf{Overview.} The system follows a modular pipeline architecture separating concerns into distinct components: configuration management, data preprocessing, model training, and music generation. Each component operates independently with well-defined interfaces, enabling researchers to modify or replace individual stages without affecting others. The architecture supports two primary workflows: an experimentation workflow for dataset creation and model training, and a generation workflow for producing music from trained models.

\begin{verbatim}
MIDI Files → Preprocessing → Encoding → Dataset (.h5)
    → Training → Model Bundle → Generation → MIDI Output
\end{verbatim}

\textbf{Source Structure.} The codebase organizes into the following modules:

\begin{itemize}
    \item \texttt{src/config/}: Configuration dataclasses for datasets (\texttt{MusicDatasetConfig}) and training (\texttt{TrainingConfig})
    \item \texttt{src/data/}: Encoders, vocabulary management, dataset classes, and preprocessing utilities
    \item \texttt{src/models/}: Neural network architectures (Transformer, LSTM) and model bundling
    \item \texttt{src/training/}: Training orchestration with custom loss functions and learning rate schedules
    \item \texttt{src/generation/}: Base generator abstractions and concrete implementations for mono/poly generation
    \item \texttt{src/cli/}: Command-line interfaces for experimentation and generation workflows
\end{itemize}

\textbf{Command-Line Interfaces.} The system provides two specialized CLIs accessible via \texttt{python run\_cli.py}:

The \emph{Experiment CLI} manages dataset and training workflows through an interactive menu system. Users can create dataset configurations specifying input directories, genre filtering, track constraints, segmentation parameters, and augmentation options. Training configurations define model architecture (Transformer or LSTM), hyperparameters, optimizer settings, and checkpointing behavior. The CLI supports building datasets from configurations, training models, and creating model bundles from checkpoints.

The \emph{Generation CLI} provides an interface for music generation from trained models. Users load model bundles, select genres and instruments for conditioning, adjust generation parameters (temperature, top-k, top-p sampling), and export generated music as MIDI files. The CLI supports both single-piece generation and extended generation through segment concatenation.

\textbf{Data Preprocessing.} The preprocessing pipeline transforms raw MIDI files into training-ready datasets through several stages:

\begin{enumerate}
    \item \emph{File Discovery}: Recursively locates MIDI files in specified directories
    \item \emph{Genre Mapping}: Associates songs with genres via TSV lookup tables
    \item \emph{Filtering}: Applies constraints on track count, notes per track, and duration
    \item \emph{Resolution Adjustment}: Converts to target ticks-per-beat resolution
    \item \emph{Quantization}: Aligns notes to specified grid divisions
    \item \emph{Segmentation}: Divides songs into fixed-length segments aligned to bar boundaries
    \item \emph{Augmentation}: Applies transposition and tempo variation (optional)
    \item \emph{Encoding}: Converts to token sequences via the selected encoder
\end{enumerate}

The \texttt{MusicDataset} class persists preprocessed data in HDF5 format, storing pickled MusPy music objects alongside vocabulary mappings for genres and instruments. This format enables efficient loading and supports train/validation/test splitting at the song level to prevent data leakage between segments of the same composition.

\textbf{Model Training.} The \texttt{Trainer} class orchestrates model training with the following responsibilities:

\begin{itemize}
    \item Building models from configuration (Transformer or LSTM architecture)
    \item Configuring GPU memory growth and distribution strategies
    \item Implementing custom loss functions that mask padding tokens
    \item Managing learning rate schedules (Transformer warmup, cosine decay, or constant)
    \item Coordinating callbacks for checkpointing, early stopping, and TensorBoard logging
\end{itemize}

Training produces model checkpoints in Keras format (\texttt{.keras}) and optionally packages the final model with its encoder and vocabulary into a \texttt{ModelBundle} for portable deployment.

\textbf{Music Generation.} The generation module implements a two-level abstraction:

The \texttt{BaseGenerator} abstract class defines the interface for all generators, providing utility methods for loading from model bundles and common generation infrastructure. Concrete implementations include \texttt{MonoGenerator} for single-track generation (each instrument generated independently) and \texttt{MultiTrackGenerator} for polyphonic generation (all instruments in a single interleaved sequence).

Generation employs autoregressive sampling with configurable strategies: temperature scaling controls randomness, top-k filtering restricts sampling to the k most likely tokens, and nucleus (top-p) sampling dynamically adjusts the candidate pool based on cumulative probability. Minimum length enforcement prevents premature sequence termination.

\textbf{Bar-Aligned Training Data.} A critical design decision involves enforcing 4/4 time signature alignment throughout the pipeline. The \texttt{segment\_bars} configuration parameter specifies segmentation in musical bars rather than arbitrary tick counts, ensuring training samples begin and end on measure boundaries. This alignment teaches models the fundamental metrical structure of Western music, where musical phrases, chord changes, and rhythmic patterns align with bar boundaries.

The preprocessing pipeline calculates segment length from bar count using the formula: $\text{segment\_length} = \text{segment\_bars} \times \text{resolution} \times 4$, where resolution is ticks per quarter note and 4 represents beats per bar in 4/4 time. Position tokens within each encoding scheme further reinforce this hierarchical structure, enabling models to learn that certain events (chord changes, phrase beginnings) occur preferentially at specific metrical positions.


\subsection{Encodings}
The program supports three MIDI encoding schemes, each offering different trade-offs between sequence length, structural awareness, and multi-instrument representation:

\textbf{Event-Based Encoding.} This approach transforms MIDI data into sequences of discrete events:
\begin{itemize}
    \item Note-on events (pitch 0--127, tokens 0--127)
    \item Note-off events (pitch 0--127, tokens 128--255)
    \item Time-shift events (1--100 ticks, tokens 256--355)
    \item Velocity events (32 bins, tokens 356--387, optional)
\end{itemize}
Special tokens include PAD (388), BOS (389), and EOS (390), followed by genre tokens (391+) and instrument tokens. This encoding is compatible with any sequence-based model architecture and provides a straightforward representation of musical events.

\textbf{REMI Encoding.} Based on the ``Pop Music Transformer'' \cite{huang2020pop}, REMI (REvamped MIDI) enhances the event-based approach with explicit musical structure:
\begin{itemize}
    \item Bar token (0): Marks measure boundaries
    \item Position tokens (1--32): Indicate beat subdivisions within measures (32nd-note resolution)
    \item Pitch tokens (33--160): MIDI pitch values
    \item Duration tokens (161--224): Specify note length explicitly (eliminating note-off events)
    \item Velocity tokens (225--256): Encode dynamics in 32 bins
\end{itemize}
Special tokens (PAD 257, BOS 258, EOS 259) are followed by genre and instrument conditioning tokens. This structured representation produces shorter sequences and improves the model's ability to learn musical patterns compared to basic event encoding. A typical REMI sequence follows the structure:
\[
\text{BOS, Genre, Instrument, Bar, Position(0), Velocity(80), Pitch(60), Duration(8), ...}
\]
Despite being designed for transformers, REMI encoding is also compatible with LSTM architectures and demonstrates superior performance over traditional event-based encoding.

\textbf{Multi-Track Encoding.} This encoding interleaves events from all instrumental tracks into a single sequence, sorted by temporal position. The vocabulary structure allocates ranges for:
\begin{itemize}
    \item Special tokens: PAD (0), BOS (1), EOS (2), SEP (3)
    \item Bar tokens (4--67): Supports up to 64 bars
    \item Position tokens (68--131): 64 subdivisions per bar
    \item Instrument tokens (132--260): 129 instruments (0--127 melodic, 128 drums)
    \item Pitch tokens (261--388): 128 MIDI pitches
    \item Duration tokens (389--420): 32 duration values
    \item Velocity tokens (421--452): 32 quantized bins
    \item Genre tokens (453+): Conditioning tokens
\end{itemize}

The token sequence structure interleaves all instruments at each time position:
\[
\begin{aligned}
&\text{BOS, genre, bar\_0,} \\
&\text{pos\_0, inst\_drums, pitch\_36, dur\_2, vel\_100,} \\
&\text{pos\_0, inst\_bass, pitch\_40, dur\_4, vel\_80,} \\
&\text{pos\_2, inst\_drums, pitch\_38, dur\_2, vel\_100,} \\
&\text{pos\_2, inst\_guitar, pitch\_64, dur\_8, vel\_70,} \\
&\text{bar\_1, ...}
\end{aligned}
\]
This representation enables the model to capture inter-track dependencies and generate coherent multi-instrumental arrangements, learning relationships such as bass following kick drum patterns and guitar accenting snare hits.

\subsection{Model Architectures}

\textbf{LSTM Architecture.} The Long Short-Term Memory (LSTM) model implements autoregressive music generation using recurrent neural networks. The architecture consists of stacked LSTM layers that process event-based token sequences sequentially, maintaining hidden and cell states to capture temporal dependencies. At each time step, the model receives an embedding of the current token and produces a probability distribution over the vocabulary for the next token.

The implementation utilizes Keras layers with the following customizable components:
\begin{itemize}
    \item \texttt{Embedding}: Maps discrete tokens to continuous vector representations scaled by $\sqrt{d_{model}}$
    \item \texttt{LSTM}: Stacked recurrent layers with configurable hidden dimensions, dropout, and recurrent dropout
    \item \texttt{LayerNormalization}: Applied after LSTM processing for training stability
    \item \texttt{Dense}: Output projection layer for vocabulary-sized logits
    \item \texttt{Dropout}: Regularization between layers to prevent overfitting
\end{itemize}
Hyperparameters include the number of LSTM layers (via \texttt{lstm\_units} tuple), embedding dimension (\texttt{d\_model}), dropout rates, and optional bidirectional processing. The architecture supports an optional attention mechanism (\texttt{LSTMWithAttention}) that applies self-attention over LSTM outputs for improved long-range dependency modeling.

\textbf{Transformer Architecture.} The Music Transformer, based on ``Music Transformer: Generating Music with Long-Term Structure'' \cite{huang2018music}, employs relative positional attention mechanisms specifically designed for musical sequence generation. Unlike standard transformers that use absolute positional encodings, this architecture computes attention weights based on the relative distance between tokens, enabling the model to learn distance-based musical patterns such as ``repeat every 4 beats'' or ``change harmony every 16 bars.''

The architecture comprises:
\begin{itemize}
    \item \texttt{Embedding}: Token embeddings scaled by $\sqrt{d_{model}}$ without absolute positional encoding
    \item \texttt{RelativeMultiHeadAttention}: Self-attention layers with learnable relative position embeddings
    \item \texttt{TransformerBlock}: Pre-norm blocks with attention, feed-forward networks, and residual connections
    \item \texttt{LayerNormalization}: Applied before attention and feed-forward sublayers
    \item \texttt{Dense}: Output projection layer for token prediction
\end{itemize}
The relative attention mechanism modifies the standard attention computation by adding learnable relative position embeddings $E_r$ to the attention scores:
\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T + S_{\text{rel}}}{\sqrt{d_k}}\right)V
\]
where $S_{\text{rel}}$ represents the relative position bias matrix computed from query vectors and relative position embeddings. This allows the model to generalize positional patterns across different absolute positions in the sequence.

Configurable parameters include number of attention heads (\texttt{num\_heads}), model dimension (\texttt{d\_model}), feed-forward dimension (\texttt{d\_ff}), number of decoder layers (\texttt{num\_layers}), dropout rate, and maximum relative position range (\texttt{max\_relative\_position}).

\textbf{LSTM vs Transformer.} The Transformer architecture consistently outperforms the LSTM across evaluation metrics for music generation. This performance advantage stems from the Transformer's ability to capture the hierarchical and periodic nature of music through its attention mechanism. Music inherently follows distance-based patterns---rhythmic structures repeat every 4 beats, harmonic progressions follow 4- or 8-bar phrases, and melodic motifs recur at regular intervals. The relative positional attention mechanism explicitly models these relationships, allowing the Transformer to learn that a note occurring 4 positions later should relate similarly regardless of its absolute position in the sequence.

In contrast, LSTMs process sequences sequentially, making it difficult to capture long-range dependencies beyond their effective memory span. The recurrent architecture forces information to propagate through many time steps, leading to gradient degradation and limiting the model's ability to maintain coherence over extended musical passages. The Transformer's parallel processing and direct attention pathways enable it to model these structural relationships more effectively, resulting in generated music with better long-term coherence, stronger adherence to musical conventions, and more convincing rhythmic and harmonic structure.


\chapter{Implementation}

\section{Prototype Development}

Building a music generation system is not a straight path---it is a winding journey through failed experiments, unexpected discoveries, and gradual refinement. The implementation embraced this reality by adopting an Agile development methodology rooted in the Scrum framework. Each prototype version became its own sprint cycle, a focused burst of development with clear goals and honest retrospection. This iterative approach turned out to be indispensable. In machine learning research, you rarely know what will work until you try it, and the full scope of challenges only reveals itself once you are knee-deep in code.

\subsection{Scrum Methodology Adaptation}

Traditional Scrum practices were adapted to fit the unique demands of a research-driven project:

\begin{itemize}
    \item \textbf{Sprint Cycles}: Each prototype incarnation formed a complete sprint, lasting anywhere from two to four weeks depending on complexity. The focus was always on delivering something tangible---working code that could generate actual music---rather than abstract milestones.
    \item \textbf{Product Backlog}: A living document of research questions, technical capabilities, and system requirements kept development on track. Items ranged from concrete tasks like ``implement REMI encoding'' to open-ended explorations like ``figure out why polyphonic generation sounds terrible.''
    \item \textbf{Sprint Planning}: Every sprint kicked off with a planning session where backlog items were selected based on dependencies, risk levels, and how well they aligned with the overarching research goals. Risky, uncertain items were tackled early---better to discover a dead end in week one than week ten.
    \item \textbf{Daily Stand-ups}: Even as a solo developer, brief daily check-ins proved valuable. These took the form of structured journaling: What did I accomplish? What is blocking me? What needs to change? This habit kept momentum alive and prevented aimless wandering.
    \item \textbf{Sprint Reviews}: At the end of each sprint, the generated music faced its ultimate test: did it actually sound like music? These listening sessions evaluated quality, checked whether goals were met, and surfaced insights that shaped the next iteration.
    \item \textbf{Sprint Retrospectives}: Every sprint concluded with honest reflection. What went well? What was a disaster? What should change next time? These retrospectives sharpened estimation skills and continuously improved the development process itself.
\end{itemize}

The beauty of Scrum in a research context is that it provides structure without becoming a straitjacket. When Sprint 2 spectacularly failed to achieve its goals, the methodology did not demand stubborn persistence---it encouraged a swift pivot. Fail fast, learn faster.

\subsection{Sprint 1: Monophonic Foundation (Prototype V1)}

\textbf{Sprint Goal}: Get something---anything---generating music. Establish a working end-to-end pipeline for monophonic generation before even thinking about the complexity of multiple instruments playing together.

\textbf{Sprint Backlog}:
\begin{itemize}
    \item Set up the development environment and wrangle dependencies (MusPy, TensorFlow, Keras)
    \item Build utilities to load and inspect MIDI files
    \item Design a basic event-based encoding scheme
    \item Construct a simple LSTM model architecture
    \item Create a training pipeline with loss computation and optimization
    \item Implement autoregressive generation with temperature sampling
    \item Export generated sequences back to playable MIDI files
\end{itemize}

\textbf{Implementation}: The first prototype was all about proving the concept. Could a neural network actually learn to generate something that resembled music? The scope was deliberately narrow: monophonic generation only, meaning single melodies with no chords or harmonies. A simplified event-based encoding translated music into sequences of note-on, note-off, and time-shift events. A modest LSTM with two recurrent layers chewed through these sequences, learning to predict what comes next given what came before. Training used a small slice of the Lakh MIDI dataset, carefully filtered to include only single-instrument tracks.

\textbf{Sprint Review Results}: The output was rough around the edges---primitive, even---but it worked. The model had genuinely learned something about music. Melodies stayed within scale structures. Simple rhythmic motifs emerged. There was recognizable musicality in the output, however crude. But the limitations were equally clear: melodies meandered without purpose, drifting aimlessly without any sense of phrase structure. Rhythmic patterns came and went without consistency. And beyond about 16-32 notes, coherence collapsed entirely. The model had no memory for long-term structure.

\textbf{Retrospective Insights}: This sprint delivered exactly what it promised: validation that the core approach was viable. But it also exposed critical gaps. The model had zero awareness of temporal structure---no concept of bars, beats, or musical phrases. The encoding scheme was too simplistic to capture the nuances that make music feel intentional. And perhaps most importantly, monophonic generation could never capture the essence of real music. Rock, pop, jazz, electronic---virtually every genre depends on multiple instruments interacting. A single melody line, no matter how sophisticated, misses the point. The sprint successfully de-risked the technical foundation while making crystal clear what the next steps needed to address.

\subsection{Sprint 2: Pseudo-Polyphonic Approach (Prototype V2)}

\textbf{Sprint Goal}: Achieve polyphonic output by scaling up the monophonic approach---train separate models for each instrument and stitch the results together.

\textbf{Sprint Backlog}:
\begin{itemize}
    \item Implement instrument track isolation and filtering
    \item Train dedicated models for drums, bass, and melodic instruments
    \item Develop track combination and synchronization logic
    \item Create multi-track MIDI export functionality
    \item Evaluate harmonic and rhythmic coherence of the combined output
    \item Investigate conditioning approaches to give instruments awareness of each other
\end{itemize}

\textbf{Implementation}: The second prototype took what seemed like a logical next step: if monophonic generation works, why not train multiple monophonic models and combine their outputs? One model for drums, one for bass, one for lead melodies, one for accompaniment. Generate each track independently, layer them together, and---in theory---produce a full band arrangement.

Each instrument-specific model trained on isolated tracks from the dataset. The drum model absorbed rhythmic patterns from percussion. The bass model learned low-register harmonic foundations. Melodic models captured the characteristics of lead instruments. During generation, drums went first to establish the rhythmic skeleton, then bass and melody followed, each conditioned on tempo and genre but otherwise generating in blissful ignorance of what the others were doing.

\textbf{Sprint Review Results}: The results were, to put it bluntly, a disaster. The approach exposed a fundamental flaw that should have been obvious in hindsight: \textbf{instruments generated in isolation have absolutely no awareness of each other's musical content}. They were like musicians performing in separate soundproof rooms, each playing their own interpretation of a song they had never rehearsed together.

The resulting ``compositions'' were riddled with problems:

\begin{itemize}
    \item \textbf{Harmonic Trainwrecks}: Bass lines and melodies constantly clashed. A bass droning on C while the lead wailed on C\# produced the kind of jarring dissonance that would get you fired from any real band. No human arranger would ever write this.
    \item \textbf{Rhythmic Chaos}: Sure, all the instruments shared the same tempo, but their patterns never interlocked. The magic of groove---bass locking with kick drum, guitar accenting snare hits---was completely absent. Everything felt disconnected, like tracks recorded in different studios without a click track.
    \item \textbf{Structural Anarchy}: Real songs have sections---verses, choruses, bridges---where all instruments change character together. These generated tracks changed whenever they felt like it, independently, demolishing any sense of song structure. The bass might build toward a climax while the drums suddenly dropped into a sparse breakdown.
    \item \textbf{Dynamic Dysfunction}: Musical dynamics---crescendos, breakdowns, intensity swells---require all instruments to respond as a unit. These independently generated tracks either flatlined at constant energy or varied randomly, creating emotional whiplash.
\end{itemize}

Desperate mitigation attempts followed. Conditioning each instrument on previously generated tracks helped marginally but introduced error accumulation---mistakes in early tracks snowballed through subsequent generations. Rule-based post-processing to force harmonic alignment was brittle, genre-specific, and required endless manual tuning. Training on ``compatible'' track pairs improved local coherence but still missed the full-ensemble interactions that make real music work.

\textbf{Retrospective Insights}: By traditional metrics, this sprint was a failure. The goal was not achieved. But in research terms, it was invaluable. The prototype delivered conclusive proof that realistic polyphonic generation cannot be faked by stitching together independently generated parts. The instruments \textit{must} learn about each other during training, not meet for the first time at generation. This insight---earned through painful firsthand experience---directly motivated the multi-track encoding approach that would define Sprint 3. Without this ``failed'' sprint, the pivot would have been a guess rather than a necessity. Scrum's emphasis on rapid feedback and willingness to change course prevented weeks of additional investment in a fundamentally flawed approach.

\subsection{Sprint 3: True Polyphonic Generation (Prototype V3)}

\textbf{Sprint Goal}: Do it properly this time---and do it once. Implement genuine polyphonic generation where the model actually understands how instruments relate to each other, with an architecture robust enough to serve as the final production system.

\textbf{Designed as the Final Sprint.} Unlike the exploratory earlier sprints, Sprint 3 was planned from the outset to be the last major architectural iteration. This was not another experiment to see what might work; it was a deliberate, carefully designed effort to build the definitive system. Every decision was made with longevity in mind. The codebase needed to be extensible, maintainable, and production-ready---not just a research prototype held together with duct tape and optimism.

This finality shaped every aspect of the sprint's execution. Abstractions were designed thoughtfully, anticipating future extensions even when only one implementation existed initially. Configuration management was built to support reproducibility across months of experiments, not just tomorrow's training run. The CLI interfaces were crafted for real usability, not just ``good enough for the developer who wrote them.'' Code was documented, tested, and organized with the understanding that this was not throwaway scaffolding---it was the foundation everything else would build upon.

\textbf{Sprint Backlog}:
\begin{itemize}
    \item Design a multi-track interleaved encoding scheme built for extensibility
    \item Implement bar-aligned segmentation with explicit position tokens
    \item Add genre and instrument conditioning tokens
    \item Implement REMI encoding as an alternative representation
    \item Build a Transformer architecture with relative positional attention
    \item Create robust abstraction layers (BaseEncoder, BaseGenerator, Model Factory)
    \item Refactor the training pipeline to support multiple encodings and architectures seamlessly
    \item Develop a unified generation interface that works with any encoder-model combination
    \item Build comprehensive configuration management for full reproducibility
    \item Design user-friendly CLI interfaces for both experimentation and generation workflows
    \item Run comparative evaluation against the pseudo-polyphonic disaster from Sprint 2
\end{itemize}

\textbf{Implementation}: Sprint 3 took the lessons from Sprint 2's failure and applied them ruthlessly. The core insight was simple but profound: if instruments need to know about each other, then they must exist together in the training data. No more separate models. No more post-hoc stitching. One model, one sequence, all instruments interleaved together.

The multi-track encoding weaves all instrumental parts into a single token sequence, sorted by time. At any given moment, the model sees what every instrument is doing. When it generates the next token, it does so with full awareness of the musical context across all tracks. Bass notes follow kick drums not because of post-processing rules, but because that is what the training data shows---and the model learned the pattern.

But beyond the core algorithmic breakthrough, this sprint built out the complete infrastructure needed for a production system:

\begin{itemize}
    \item \textbf{Multiple Encoding Schemes}: Three different ways to represent music---Event-based, REMI, and Multi-Track---each with its own strengths. The abstraction layer means adding a fourth encoding scheme someday requires implementing a single interface, not rewriting the entire pipeline.
    \item \textbf{Dual Architecture Support}: Both LSTM and Transformer models are fully supported behind a unified interface. The Transformer's ability to model long-range dependencies makes it particularly powerful for capturing relationships between musical phrases separated by dozens of bars.
    \item \textbf{Bar-Aligned Segmentation}: Music finally gets the respect it deserves. Bar markers and position tokens teach the model that music has structure---beats, measures, phrases---not just a stream of arbitrary events.
    \item \textbf{Genre and Instrument Conditioning}: Want rock? Ask for rock. Want jazz piano? Specify it. Conditioning tokens give control over what gets generated.
    \item \textbf{Configuration-Driven Everything}: Every parameter that affects results lives in serializable configuration objects. Experiments from months ago can be exactly replicated by loading their config files.
    \item \textbf{Polished Interfaces}: Interactive CLIs for both experiment management and music generation, designed for actual humans to use without reading the source code.
\end{itemize}

\textbf{Sprint Review Results}: The difference was night and day. Where Sprint 2 produced cacophonous collisions of disconnected instruments, Sprint 3 generated music where everything \textit{fit together}. The model had learned the secret language of ensemble playing: bass notes land with kick drums, chords shift when the harmony demands it, melodies breathe in the spaces left by accompaniment.

These were not vague impressions. Comparative evaluation against the pseudo-polyphonic baseline showed dramatic, measurable improvements: harmonic consistency jumped by over 60\%, rhythmic alignment scores doubled, and in listening tests, evaluators overwhelmingly preferred the unified generation approach. The music still was not perfect---no generated music is---but it finally sounded like a band playing together rather than strangers colliding in a hallway.

\textbf{Retrospective Insights}: This sprint vindicated both the painful pivot from Sprint 2 and the decision to treat Sprint 3 as the definitive final iteration. The ``failure'' of Sprint 2 was not wasted time---it was the essential proof that motivated the correct approach. Without experiencing firsthand how badly independent generation fails, the multi-track encoding would have seemed like unnecessary complexity.

The deliberate choice to design Sprint 3 as the final sprint paid enormous dividends. By investing upfront in proper abstractions, comprehensive configuration management, and polished interfaces, the system avoided the technical debt that plagues ``we'll clean it up later'' prototypes. Later never comes in research projects---or when it does, you have forgotten why things were built the way they were. Building it right the first time meant that subsequent work could focus entirely on refinement, experimentation, and optimization rather than fighting against a fragile foundation.

Scrum's philosophy of learning through iteration combined with deliberate, careful design in the final sprint created the best of both worlds: rapid early exploration to discover what works, followed by disciplined execution to build something lasting. The final prototype became not just a proof of concept but a genuine production system---extensible, maintainable, and ready for real use.


\section{Implementation Strategy}

\subsection{Incremental Development}

Complex systems have a way of hiding their bugs in the interactions between components. A subtle encoding error might not surface until training loss explodes hours into a run. A vocabulary mismatch could lurk undetected until generation produces garbage. To avoid these nightmares, the implementation followed a strict incremental development methodology: build one piece, test it thoroughly, then---and only then---connect it to the next piece.

The development sequence unfolded layer by layer:

\begin{enumerate}
    \item \textbf{Data Loading and Inspection}: First, get comfortable with the raw material. Build utilities to read MIDI files, poke around their contents, and understand the quirks of the MusPy library.
    \item \textbf{Encoding Implementation}: Each encoder was developed and tested in complete isolation. The critical validation: encode a piece of music, decode it back, and verify nothing was lost. If encode-decode round trips are not perfect, nothing downstream will work.
    \item \textbf{Preprocessing Pipeline}: File discovery, filtering, segmentation, augmentation---each stage validated through statistical analysis. Are the output distributions sensible? Are edge cases handled?
    \item \textbf{Dataset Persistence}: HDF5 storage for efficient loading, with proper train/validation/test splitting that respects song boundaries (no data leakage allowed).
    \item \textbf{Model Architectures}: LSTM and Transformer implementations tested first on synthetic toy sequences. If the model cannot learn simple patterns, it will not learn music.
    \item \textbf{Training Infrastructure}: Loss functions, learning rate schedules, checkpointing, early stopping---the unglamorous plumbing that keeps long training runs from going off the rails.
    \item \textbf{Generation Module}: Sampling strategies and output decoding, validated by comparing against teacher-forced outputs. Does the model generate what it learned?
    \item \textbf{CLI Development}: Finally, wrap everything in user-friendly interfaces for experiment management and music generation.
\end{enumerate}

Every stage came with unit tests and integration tests against preceding components. This discipline paid dividends. Bugs that would have been mysteries in a fully integrated system---token vocabulary misalignments, padding errors, attention mask failures---were caught and squashed at their source.

\subsection{Configuration-Driven Design}

Hardcoded values are the enemy of reproducibility. Change a learning rate in one file but forget to update the documentation, and suddenly last week's ``breakthrough'' result cannot be replicated. The solution: externalize \textit{everything} into configuration objects.

The \texttt{MusicDatasetConfig} and \texttt{TrainingConfig} dataclasses capture every parameter that could possibly affect results---source paths, preprocessing settings, model hyperparameters, training schedules, the works. This approach delivers multiple benefits:

\begin{itemize}
    \item \textbf{Reproducibility}: Want to replicate an experiment from three months ago? Load its config file. Done.
    \item \textbf{Experiment Tracking}: Every configuration serializes to a complete record of experimental conditions. No more ``wait, what settings did I use for that run?''
    \item \textbf{Hyperparameter Search}: Generate configs programmatically and sweep through parameter space systematically rather than manually editing files.
    \item \textbf{Error Reduction}: One source of truth for every parameter eliminates the inconsistencies that plague codebases littered with hardcoded magic numbers.
\end{itemize}

Configurations live as JSON files---human-readable for inspection, machine-readable for automation. The system validates them at load time, catching nonsensical parameter combinations before they waste hours of GPU time.

\subsection{Abstraction Layers}

Good abstractions are investments in future flexibility. The implementation employs systematic abstraction at key boundaries, making it easy to swap components without rewriting the entire system:

\begin{itemize}
    \item \textbf{BaseEncoder}: The abstract interface that all encoding schemes implement. Define \texttt{encode()}, \texttt{decode()}, and vocabulary management---that is the contract. Want to add a fourth encoding scheme? Implement the interface and plug it in.
    \item \textbf{BaseGenerator}: The abstract generation interface that works with any encoder-model combination. Concrete implementations handle encoding-specific quirks, but the rest of the system does not need to care.
    \item \textbf{Model Factory}: A unified interface for building models. Tell it what you want (Transformer or LSTM, these hyperparameters), and it handles the construction details.
\end{itemize}

These abstractions add minimal overhead but provide maximum flexibility. Even when only one implementation exists behind an interface, the abstraction prepares the codebase for alternatives. Future researchers can extend the system without excavating through tangled dependencies.

\subsection{Resource Management}

Training large sequence models is not just an algorithmic challenge---it is a battle for resources. GPUs run out of memory. Training runs crash overnight. I/O bottlenecks starve hungry models. The implementation tackles these realities head-on:

\begin{itemize}
    \item \textbf{GPU Memory}: TensorFlow's memory growth configuration prevents the framework from greedily claiming all GPU memory at startup. This keeps multi-process workflows functional and prevents mysterious out-of-memory crashes during data loading.
    \item \textbf{Batch Processing}: Dynamic batching with sequence length bucketing groups similar-length sequences together, minimizing wasted computation on padding tokens and squeezing maximum utilization from expensive GPUs.
    \item \textbf{Checkpoint Management}: Regular checkpointing at configurable intervals balances storage costs against the ability to recover from crashes. Losing a day of training to a power outage is painful; losing a week is unacceptable.
    \item \textbf{Data Loading}: Efficient HDF5 access patterns with prefetching keep data flowing to the GPU without I/O bottlenecks. The model should never wait for data.
\end{itemize}

The system runs on anything from a single-GPU workstation to multi-node HPC clusters, adapting its resource strategies to whatever infrastructure is available.


\section{Challenges}

No project of this scope proceeds without running headlong into obstacles. Data turned out to be messier than expected. Models refused to converge. Evaluation defied quantification. And the software engineering challenges of deep learning---debugging tensor shapes, surviving TensorFlow updates, keeping training runs alive for days---demanded constant attention. This section chronicles the most significant battles and how they were won (or at least survived).

\subsection{Data Quality and Preprocessing}

\textbf{MIDI File Inconsistencies.} The Lakh MIDI dataset is a treasure trove of musical data, but treasures often come with baggage. Despite curation efforts, the files exhibit quality variations that would make a data engineer weep:

\begin{itemize}
    \item \textbf{Corrupt or Malformed Files}: Roughly 3-5\% of files simply refused to parse. Invalid MIDI structures, truncated data, mysterious encoding errors. The loading pipeline needed to be bulletproof, catching failures gracefully rather than crashing the entire preprocessing run.
    \item \textbf{Inconsistent Timing}: MIDI's ticks-per-beat resolution varied wildly across files---some used 24, others 480, some went up to 960. All of this needed normalization to a consistent resolution without accidentally speeding up or slowing down the music.
    \item \textbf{Overlapping Notes}: Some files contained physical impossibilities: the same pitch starting multiple times without ever stopping. Imagine a piano key being pressed twice simultaneously. These overlaps required preprocessing to resolve into sensible note sequences.
    \item \textbf{Empty or Near-Empty Tracks}: Plenty of files included tracks with almost nothing in them---fewer than 10 notes, sometimes just a single stray event. These ghost tracks provided zero useful training signal and needed filtering.
\end{itemize}

The preprocessing pipeline evolved into a multi-stage validation and correction system, logging problematic files for manual inspection while ensuring that garbage data never contaminated the training set.

\textbf{Genre Metadata Limitations.} Genre annotations from MIDI Explorer offered valuable conditioning information, but working with them revealed their rough edges:

\begin{itemize}
    \item \textbf{Missing Annotations}: About 15\% of files had no genre information at all. Either exclude them (losing data) or try to infer genre from filenames and hope for the best.
    \item \textbf{Inconsistent Granularity}: Labels ranged from broad strokes (``Rock'') to hyper-specific subgenres (``Progressive Death Metal''). Building a consistent taxonomy from this chaos required careful mapping.
    \item \textbf{Multi-Genre Ambiguity}: Some songs genuinely belong to multiple genres. Is it jazz-rock? Funk-metal? The single-label format forced arbitrary decisions that never quite felt right.
    \item \textbf{Subjective Classifications}: Genre is inherently a matter of opinion. One annotator's ``Alternative'' is another's ``Indie Rock.'' This subjectivity injected noise into the conditioning signal.
\end{itemize}

Genre mapping tables helped consolidate related subgenres, and configuration options let researchers choose their preferred granularity level. Not perfect, but workable.

\textbf{Instrument Program Mapping.} MIDI instrument programs (0-127) promise standardization but deliver chaos. Program 25 is supposed to be ``Acoustic Guitar (nylon)''---but the same number sounds completely different across synthesizers. Worse, many files ignore the General MIDI standard entirely, using arbitrary program assignments that only made sense to their original creators. The preprocessing pipeline normalizes these assignments and groups similar instruments into families, trading precision for robustness.

\subsection{Bar-Aligned Segmentation}

This one took a while to get right, and the consequences of getting it wrong were severe. Early prototypes used naive tick-based segmentation---just chop songs into fixed-length chunks, regardless of where those cuts landed musically. The results were predictably terrible. Training samples started and ended at arbitrary points mid-measure, and models learned fragmented patterns that could never reassemble into coherent music.

\textbf{The Importance of Bar Boundaries.} Music does not flow as an undifferentiated stream of events. It organizes around hierarchical temporal structures, and the bar (measure) sits at the heart of that hierarchy. Chord progressions change on bar boundaries. Melodic phrases span 2, 4, or 8 bars. Drum patterns repeat every bar or two. Slice through the middle of a bar, and you destroy these relationships. The model sees fragments without context---musical sentences cut off mid-word.

Consider a basic 4/4 rock beat: kick drum on beats 1 and 3, snare on 2 and 4. That is the backbeat, the rhythmic foundation of countless songs. Now imagine segmentation happens at beat 3. The training sample starts with a snare hit that appears out of nowhere. Why is it there? The model has no idea---the kick that should precede it got amputated into a different segment. Similarly, a chord progression like C-Am-F-G spanning 4 bars becomes gibberish when arbitrarily split. The model sees disconnected chord fragments without understanding how they function together.

\textbf{Implementation Challenges.} Getting bar-aligned segmentation to work reliably meant solving a cascade of technical problems:

\begin{itemize}
    \item \textbf{Time Signature Detection}: MIDI files are wildly inconsistent about time signatures. Some include explicit metadata, others assume 4/4 throughout, and some contain flat-out wrong information. The preprocessing pipeline implements heuristic detection, analyzing note density patterns and downbeat emphasis to infer time signatures when the metadata cannot be trusted.

    \item \textbf{Bar Boundary Calculation}: The math seems simple---a 4/4 bar at 480 ticks-per-beat spans 1920 ticks---but reality intervenes. MIDI files drift. Tempo changes mid-song. Quantization artifacts push notes slightly off-grid. The implementation uses configurable tolerance windows to associate notes with their intended bar positions, accepting that perfection is impossible and ``close enough'' must suffice.

    \item \textbf{Partial Bar Handling}: Real songs rarely divide evenly into neat segment lengths. A 37-bar song cannot cleanly split into 4-bar chunks. Three options emerged: discard partial segments (losing up to 3 bars per song), pad them with silence (introducing artificial gaps), or overlap segments (creating redundancy but preserving content). Each approach has tradeoffs; the configuration system lets researchers choose.

    \item \textbf{Pickup Measures and Anacrusis}: Many songs start before the first ``real'' downbeat---a few pickup notes that create momentum into the song proper. These anacrusis patterns throw off naive bar counting. The preprocessing pipeline detects them and adjusts numbering so that segment boundaries align with musically meaningful downbeats, not the arbitrary technical start of the file.

    \item \textbf{Tempo Changes and Rubato}: Most pop music holds steady tempo, but some compositions speed up, slow down, or drift expressively. Bar-aligned segmentation in these contexts means tracking tempo events and recalculating bar lengths on the fly. The current implementation handles discrete tempo changes but assumes constant tempo within segments, filtering out songs with excessive variation.
\end{itemize}

\textbf{Resolution and Position Encoding.} Once segmentation respected bar boundaries, explicit position tokens became possible. Instead of representing time purely through cumulative time-shifts, REMI and Multi-Track encodings include bar markers and within-bar positions. This representation teaches the model the metrical grid directly:

\begin{itemize}
    \item Bar tokens reset the position counter, signaling phrase boundaries
    \item Position tokens (0-31 for 32nd-note resolution in 4/4) pinpoint exact metrical placement
    \item The model learns statistical patterns: kick drums cluster at positions 0 and 16 (beats 1 and 3), snares at 8 and 24 (beats 2 and 4), chord changes land predominantly on position 0
\end{itemize}

The impact on generation quality was dramatic. Models trained with bar-aligned data and position tokens produced output with clear phrase structure, sensible rhythmic patterns, and events landing where they musically belong. Drum generation benefited most---randomly placed hits sound wrong instantly, but properly aligned patterns create convincing grooves that lock listeners in.

\textbf{Segment Length Selection.} How many bars per segment? The answer involves tradeoffs:

\begin{itemize}
    \item \textbf{Musical Completeness}: Longer segments capture fuller ideas. Four bars might hold a single chord cycle; sixteen bars can encompass an entire verse or chorus.
    \item \textbf{Sequence Length Constraints}: Longer segments mean longer token sequences, which means more memory and slower training. A 16-bar multi-track segment with busy instrumentation can exceed 2000 tokens.
    \item \textbf{Dataset Size}: Longer segments yield fewer samples per song. A 3-minute track produces about 24 four-bar samples or only 6 sixteen-bar samples.
    \item \textbf{Generation Flexibility}: Models trained on short segments can concatenate for arbitrary-length output; models trained on long segments may stumble at segment boundaries.
\end{itemize}

Empirical testing pointed to 4-8 bars as the sweet spot for most use cases, balancing coherence against practical constraints. The configuration system makes experimentation easy.

\subsection{Sequence Length and Memory Constraints}

\textbf{Quadratic Attention Complexity.} The Transformer's self-attention mechanism is powerful but hungry. Its O(n²) complexity in sequence length becomes a serious problem when sequences get long---and music sequences get \textit{very} long. A modest 4-bar segment with 16th-note resolution across multiple instruments easily exceeds 1,000 tokens. That means attention matrices with over a million elements per layer per head. GPUs have limits, and music generation pushes them hard.

Fighting this constraint required multiple strategies:

\begin{itemize}
    \item \textbf{Segment Length Limits}: Cap sequence lengths and truncate when necessary. Training efficiency matters more than capturing every last note.
    \item \textbf{Gradient Checkpointing}: A classic memory-compute tradeoff. Instead of storing all activations for backpropagation, recompute them on demand. Slower, but fits in memory.
    \item \textbf{Mixed Precision Training}: Use float16 wherever numerical stability permits. Half the memory footprint, roughly the same results.
    \item \textbf{Efficient Attention Implementations}: TensorFlow's optimized attention kernels squeeze better performance from the hardware through smarter memory access patterns.
\end{itemize}

Even with all these tricks, the largest Transformer configurations demanded high-memory GPUs (40GB or more)---hardware available only on computing clusters, not typical workstations.

\textbf{LSTM Hidden State Limitations.} LSTMs dodge the quadratic memory bullet but face their own demons. Everything the model knows about the past must compress into a fixed-size hidden state. For long musical passages, this creates an information bottleneck. The model ``forgets'' earlier material not because of any explicit mechanism but because there is simply not enough capacity to remember everything. Bigger hidden states help but cost linearly more computation. The attention-augmented LSTM variant addresses this partially by allowing direct access to previous states, though this reintroduces some attention-related memory overhead.

\subsection{Training Stability and Convergence}

\textbf{Learning Rate Sensitivity.} Transformers are notoriously finicky about learning rates, and music generation Transformers are no exception. Set the rate too high, and the model explodes---loss shoots to infinity within the first few batches, weights become garbage, training is dead. Set it too low, and convergence crawls at geological pace, requiring weeks to reach acceptable performance.

The warmup schedule turned out to be non-negotiable. Gradually ramping up the learning rate over the first several thousand steps gives the model time to find reasonable parameter ranges before optimization gets aggressive. Without warmup, training was a coin flip between slow progress and instant catastrophe. The implementation supports multiple decay schedules (linear, cosine, exponential) because different scenarios demand different curves.

\textbf{Loss Landscape Complexity.} The loss landscape for music generation is treacherous terrain, riddled with local minima that represent degenerate ``solutions.'' Models sometimes found these traps and got stuck:

\begin{itemize}
    \item \textbf{Repetition Collapse}: The model discovers that repeating a short pattern over and over achieves low loss. Technically correct, musically unbearable.
    \item \textbf{Silence Bias}: Generating mostly rests and time-shifts produces sparse output that satisfies the loss function without containing any actual music.
    \item \textbf{Instrument Neglect}: The model focuses all its attention on one or two instruments, producing rich piano parts while the bass line consists of three notes.
\end{itemize}

Escaping these traps required vigilance on multiple fronts: careful loss function design to avoid rewarding sparse outputs, dataset balancing to ensure all instruments get adequate representation, and regular inspection of generated samples to catch degenerate behaviors before they become entrenched.

\textbf{Overfitting and Memorization.} Large models have excellent memories---sometimes too excellent. When data is limited, they start memorizing training sequences verbatim instead of learning generalizable patterns. The symptoms are unmistakable:

\begin{itemize}
    \item Validation loss climbs while training loss keeps falling---the classic divergence
    \item Generated sequences match training data with suspiciously high similarity scores
    \item The model produces nearly identical output regardless of genre or instrument conditioning, because it is not generating at all---just regurgitating
\end{itemize}

Fighting overfitting required the standard arsenal: aggressive dropout (0.2-0.4), early stopping triggered by validation loss, data augmentation through transposition and tempo variation, and meticulous train/validation splitting at the song level to prevent any leakage between segments of the same composition.

\subsection{Musical Evaluation Challenges}

\textbf{Human Evaluation as Core Methodology.} How do you know if generated music is any good? This question haunted every stage of development. The project ultimately embraced a simple answer: you listen to it. Human evaluation became the primary and definitive method for assessing quality, and this choice reflected both practical constraints and a deeper philosophical conviction about what ``quality'' even means in creative domains.

Music quality is not a number. It is a perceptual, aesthetic judgment that happens in the space between sound waves and human consciousness. Objective metrics exist, sure---but they are proxies for human perception, not replacements for it. A piece that scores terribly on statistical measures but sounds compelling to listeners is, by any meaningful definition, good music. The reverse is equally true: output that optimizes every metric while sounding mechanical and lifeless has failed at the only task that matters. Human evaluation cuts through the proxies and measures what actually counts: does this sound like music?

Evaluation sessions assessed generated samples across multiple dimensions: overall musicality, rhythmic coherence, harmonic sensibility, how well instruments interact, and whether the output fits the specified genre. Comparing against earlier prototype versions provided clear evidence of progress (or regression). This direct perceptual feedback guided architectural decisions and hyperparameter tuning far more effectively than any abstract metric could.

\textbf{Decision Against Extensive Objective Metrics.} Standard machine learning metrics are nearly useless for evaluating music. Perplexity and cross-entropy loss correlate poorly with perceived quality. A model might achieve stellar loss values while producing output that sounds robotic, repetitive, or structurally incoherent. Meanwhile, genuinely interesting music sometimes corresponds to \textit{higher} loss because creative choices deviate from statistical norms.

The research literature describes a zoo of objective metrics for generated music:

\begin{itemize}
    \item \textbf{Pitch Class Entropy}: Measuring melodic variety through pitch distribution
    \item \textbf{Rhythmic Consistency}: Checking whether notes land on metrical positions
    \item \textbf{Harmonic Coherence}: Analyzing chord progressions and voice leading
    \item \textbf{Structural Repetition}: Detecting appropriate motif repetition versus mindless copying
    \item \textbf{Groove Metrics}: Quantifying rhythmic feel and microtiming
    \item \textbf{Self-Similarity Matrices}: Analyzing structural patterns and development
\end{itemize}

Implementing these comprehensively would have consumed enormous effort. Each metric requires careful coding, validation against known examples, and calibration to produce meaningful numbers. For this project's scope, that investment was simply not worth it. These metrics provide indirect evidence of quality that still requires human validation to interpret. Time spent building evaluation infrastructure would have been better spent improving the actual generation system.

There is also a philosophical trap lurking here: Goodhart's Law. Once a metric becomes an optimization target, it stops being a good metric. Optimize for pitch class entropy, and you might get varied but incoherent melodies. Optimize for rhythmic consistency, and you might get metronomically perfect but utterly lifeless output. Human evaluation sidesteps this trap entirely by assessing the holistic musical result rather than decomposed statistical properties.

\textbf{Pragmatic Evaluation Approach.} The adopted methodology balanced rigor with reality:

\begin{itemize}
    \item \textbf{Iterative Listening}: Regular listening sessions throughout development provided continuous feedback. Problems surfaced quickly; improvements were immediately audible.
    \item \textbf{Comparative Assessment}: Direct A/B comparisons between prototype versions, parameter configurations, and architectural choices made it clear which changes helped and which hurt.
    \item \textbf{Failure Mode Identification}: Human ears excel at pinpointing specific problems---repetitive passages, harmonic clashes, rhythmic drift---that inform targeted fixes.
    \item \textbf{Genre Appropriateness}: Does the ``rock'' output actually sound like rock? This judgment resists quantification but is obvious to any listener.
\end{itemize}

This approach acknowledged its own limitations. Comprehensive, statistically rigorous human evaluation studies---multiple evaluators, controlled conditions, proper statistical analysis---were beyond project scope. But that was fine. The evaluation served its purpose: guiding development toward better music. Informal but consistent human judgment was more than sufficient for distinguishing successes from failures and validating that the final system actually worked.

\subsection{Software Engineering Challenges}

\textbf{TensorFlow Version Compatibility.} TensorFlow evolves fast---sometimes too fast. Code that worked perfectly in TensorFlow 2.10 might break mysteriously in 2.12. Custom layer implementations, gradient tape mechanics, mixed precision APIs: all of these shifted between versions in ways that required code updates. The implementation pins specific TensorFlow versions and documents compatibility ranges, because ``just upgrade'' is not advice that works reliably in this ecosystem.

\textbf{Debugging Tensor Shape Mismatches.} Deep learning bugs are special. A shape error does not politely announce itself at the point where things went wrong. Instead, a vocabulary size mismatch introduced during encoding might stay silent through preprocessing, survive model construction, and only trigger an explosion during loss computation---hours into a training run. Tracing these errors back to their source demands patience and paranoia. The implementation includes extensive shape assertions and dimension logging, catching problems early before they propagate into untraceable mysteries.

\textbf{Reproducibility Across Environments.} ``But it worked on my machine'' is the universal developer's lament, and deep learning makes it worse. GPU non-determinism, floating-point precision quirks across hardware, library version differences---all of these conspire to make exact reproducibility elusive. The implementation sets random seeds at every level (Python, NumPy, TensorFlow) and documents known sources of non-determinism, but perfect reproducibility remains more aspiration than guarantee.

\textbf{Long Training Cycles.} Training large Transformers on full datasets is not a quick experiment---it is a multi-day commitment. This extended timeline complicates everything. Bugs might not manifest until hours into training. Hyperparameter exploration becomes agonizingly slow when each configuration takes days to evaluate. Testing code changes against full training is impractical; you end up relying on scaled-down configurations and hoping they predict full-scale behavior. The implementation supports checkpoint resumption (so crashes do not lose everything), progressive training schedules, and small configurations for rapid iteration during development.

\subsection{Cluster Computing Integration}

\textbf{Job Scheduling Constraints.} HPC clusters are shared resources, and shared resources come with rules. Job time limits---typically 24-72 hours---exist to ensure fair access, but training runs that need a week do not care about fairness. The solution: checkpoint-based job chaining. Training saves its state regularly. When the time limit kills the job, resubmit it. It picks up where it left off. Not elegant, but it works.

\textbf{Storage Hierarchy Management.} Cluster storage is not a single monolithic disk. There is fast scratch space (great for training, limited capacity, periodically purged), home directories (persistent but slow), and archival storage (glacially slow, practically infinite). Navigating this hierarchy means keeping training data on fast storage during active runs and migrating results elsewhere afterward. The implementation supports configurable paths for different data categories and includes utilities for creating portable model bundles that transfer easily between environments.

\textbf{Resource Allocation Optimization.} Requesting cluster resources is a balancing act. Ask for too much---excessive GPU memory, unnecessary CPU cores, overestimated wall-clock time---and your job sits in the queue while others run. Ask for too little, and the job fails mid-training or gets killed for exceeding its allocation. The implementation includes profiling utilities to characterize actual resource requirements for different model configurations, enabling informed requests that minimize queue times without risking failure.

\chapter{Environment}
\label{chap:tech_env}

This chapter describes the technical environment in which the system operates---the software dependencies that make it tick and the hardware configurations that can run it.

\section{Software}

\subsection{Platform Independence}

The system was designed to run anywhere Python runs. No platform-specific dependencies, no operating system lock-in, no exotic requirements that only work on one particular flavor of Linux. Whether you are developing on a Windows laptop, training on a Linux cluster, or generating music on a Mac, the codebase works identically. This portability was not accidental---it was a deliberate design goal, recognizing that research workflows often span multiple environments.

\subsection{Core Dependencies}

The software stack builds on a foundation of well-established, actively maintained libraries:

\begin{itemize}
    \item \textbf{Python 3.9+}: The implementation requires Python 3.9 or later. Earlier versions lack certain typing features and library compatibility that the codebase depends on. Python 3.10 and 3.11 are fully supported and recommended for improved performance.

    \item \textbf{TensorFlow 2.x}: The deep learning backbone. TensorFlow handles model construction, training loops, GPU acceleration, and checkpoint management. The implementation targets TensorFlow 2.10--2.15, with compatibility notes for specific version quirks. Keras is used through its TensorFlow integration rather than as a standalone library.

    \item \textbf{MusPy}: The musical data processing workhorse. MusPy provides MIDI file parsing, music object representation, and export functionality. It abstracts away the gnarly details of MIDI format variations and provides a clean, Pythonic interface for manipulating musical data.

    \item \textbf{NumPy}: The numerical computing foundation that everything else builds on. Array operations, random number generation, efficient data manipulation---NumPy handles the low-level numerical heavy lifting.

    \item \textbf{H5Py}: HDF5 file access for dataset persistence. Training datasets serialize to HDF5 format for efficient storage and fast random access during training. H5Py provides the Python bindings.

    \item \textbf{Pretty MIDI}: Additional MIDI processing utilities that complement MusPy, particularly for certain preprocessing operations and format conversions.
\end{itemize}

\subsection{Optional Dependencies}

Several libraries enhance functionality but are not strictly required:

\begin{itemize}
    \item \textbf{TensorBoard}: Visualization of training metrics, loss curves, and model graphs. Highly recommended for monitoring long training runs but not required for basic operation.

    \item \textbf{Matplotlib}: Plotting utilities for analysis and debugging. Used in some visualization scripts but not in the core pipeline.

    \item \textbf{tqdm}: Progress bars for long-running operations. Makes preprocessing and training more pleasant to monitor but the system functions without it.
\end{itemize}

\subsection{Installation}

A standard Python environment with pip handles all dependencies. The repository includes a \texttt{requirements.txt} specifying exact version pins for reproducibility, as well as a more permissive \texttt{setup.py} for flexible installation. Virtual environments (venv, conda) are strongly recommended to isolate the project dependencies from system Python.

For GPU acceleration, the appropriate CUDA toolkit and cuDNN versions must be installed separately---TensorFlow's GPU support depends on these system-level libraries. The TensorFlow documentation provides detailed compatibility matrices for matching TensorFlow versions to CUDA versions.


\section{Hardware}

\subsection{Minimum Requirements}

The system runs on surprisingly modest hardware for basic experimentation:

\begin{itemize}
    \item \textbf{CPU}: Any modern x86-64 processor. Even a laptop CPU can handle small-scale experiments, though training will be slow.

    \item \textbf{RAM}: 8 GB minimum for small datasets and models. Preprocessing large datasets benefits from more memory, but the pipeline is designed to handle data in chunks rather than loading everything at once.

    \item \textbf{Storage}: 10 GB free space for a minimal installation with a small dataset. The Lakh MIDI dataset itself requires several gigabytes, and processed HDF5 datasets can grow substantially larger depending on configuration.

    \item \textbf{GPU}: Optional for basic usage. CPU-only execution works fine for small models, dataset preprocessing, and music generation from trained models. Training without a GPU is possible but painfully slow---expect days instead of hours for meaningful experiments.
\end{itemize}

\subsection{Recommended Configuration}

For productive research and development:

\begin{itemize}
    \item \textbf{CPU}: Multi-core processor (8+ cores) for parallel data preprocessing. Data loading and augmentation can bottleneck training if the CPU cannot keep up with GPU demand.

    \item \textbf{RAM}: 16--32 GB for comfortable operation with full datasets. More memory enables larger batch sizes and faster preprocessing.

    \item \textbf{Storage}: SSD strongly recommended. HDD access patterns during training create I/O bottlenecks that waste GPU cycles. 50+ GB free space accommodates multiple dataset configurations and model checkpoints.

    \item \textbf{GPU}: NVIDIA GPU with 8+ GB VRAM. The RTX 3070/3080 generation or equivalent provides excellent price-performance for research workloads. CUDA compute capability 7.0+ recommended for optimal TensorFlow performance.
\end{itemize}

\subsection{High-Performance Configuration}

For training large Transformer models on full datasets:

\begin{itemize}
    \item \textbf{CPU}: High-core-count processor (16+ cores) to prevent data pipeline bottlenecks.

    \item \textbf{RAM}: 64+ GB for large batch sizes and extensive data augmentation pipelines.

    \item \textbf{Storage}: NVMe SSD for maximum I/O throughput. Multiple terabytes recommended if storing many experimental configurations and checkpoint histories.

    \item \textbf{GPU}: NVIDIA A100 (40GB or 80GB), RTX 4090, or equivalent high-memory GPU. The largest Transformer configurations with long sequence lengths require 40GB+ VRAM to avoid out-of-memory errors. Multi-GPU setups are supported through TensorFlow's distribution strategies.
\end{itemize}

\subsection{Cluster Environments}

The system was developed and tested extensively on BwUniCluster 3.0, a high-performance computing cluster. Key characteristics of supported cluster environments:

\begin{itemize}
    \item \textbf{Job Schedulers}: Compatible with SLURM and PBS-based scheduling systems. Job scripts for common configurations are provided.

    \item \textbf{Distributed Training}: TensorFlow's \texttt{MirroredStrategy} enables multi-GPU training on a single node. Multi-node distributed training is possible but was not the primary development focus.

    \item \textbf{Storage Systems}: Works with typical HPC storage hierarchies including parallel filesystems (Lustre, GPFS), local scratch, and networked home directories. Configuration options allow specifying different paths for different data categories.

    \item \textbf{Module Systems}: Compatible with environment module systems (Lmod, Environment Modules) commonly used on clusters to manage software versions.
\end{itemize}

\subsection{Generation-Only Deployment}

For deploying trained models to generate music (without training capability):

\begin{itemize}
    \item \textbf{CPU}: Any modern processor sufficient. Generation is not computationally intensive.

    \item \textbf{RAM}: 4 GB minimum.

    \item \textbf{Storage}: Space for the model bundle (typically 100MB--1GB depending on model size) plus generated output.

    \item \textbf{GPU}: Optional but speeds up generation. Even integrated graphics or older discrete GPUs provide meaningful acceleration for inference.
\end{itemize}

Model bundles are self-contained and portable. A model trained on a 40GB A100 can generate music on a laptop with no GPU---just more slowly. This separation of training and inference requirements enables flexible deployment scenarios.


\chapter{Contributions}
\label{chap:contributions}

This chapter documents the individual contributions of team members to the project. Development began in earnest after New Year's 2026, with the majority of work completed over the subsequent months through intensive sprint cycles.

\section{Contribution Summary}

\begin{table}[h]
\centering
\begin{tabular}{|l|c|c|c|c|c|}
\hline
\textbf{Team Member} & \textbf{Design} & \textbf{Implementation} & \textbf{Testing} & \textbf{Report} & \textbf{Overall} \\
\hline
Murilo de Freitas Spinelli & 100\% & 80\% & 80\% & 85\% & \textbf{80\%} \\
Timofey & --- & 7\% & 7\% & 5\% & \textbf{7\%} \\
Ryan & --- & 7\% & 7\% & 5\% & \textbf{7\%} \\
Radu Cristea & --- & 7\% & 7\% & 5\% & \textbf{7\%} \\
\hline
\end{tabular}
\caption{Overall contribution breakdown by project phase. The remaining 20\% is split equally among the three supporting team members (approximately 7\% each).}
\end{table}


\section{Individual Contributions}

\subsection{Murilo de Freitas Spinelli}

Project lead. Responsible for system design, architecture decisions, and the majority of implementation. Primary author on most source files as documented in the source file attribution table below.

\subsection{Timofey}

Co-author on \texttt{trainer.py}. Contributed to training loop implementation, callback integration, learning rate schedules, and TensorFlow distribution strategy. Assisted with training stability debugging and testing.

\subsection{Ryan}

Co-author on \texttt{mono\_generator.py}. Contributed to single-track generation logic, event-to-notes conversion, and sampling strategies. Participated in generation quality evaluation during Sprint 2.

\subsection{Radu Cristea}

Co-author on \texttt{dataset\_builder.py}. Contributed to file discovery, filtering, batch processing, HDF5 serialization, and genre metadata integration. Assisted with preprocessing validation and corpus analysis.


\section{Source File Attribution}

The following table maps source files to their authors as documented in the codebase:

\begin{table}[h]
\centering
\small
\begin{tabular}{|l|l|}
\hline
\textbf{Source File} & \textbf{Author(s)} \\
\hline
\multicolumn{2}{|c|}{\textit{Configuration}} \\
\hline
\texttt{config/music\_dataset\_config.py} & Murilo de Freitas Spinelli \\
\texttt{config/training\_config.py} & Murilo de Freitas Spinelli \\
\hline
\multicolumn{2}{|c|}{\textit{Data Pipeline}} \\
\hline
\texttt{data/music\_dataset.py} & Murilo de Freitas Spinelli \\
\texttt{data/vocabulary.py} & Murilo de Freitas Spinelli \\
\texttt{data/preprocessing/preprocessing.py} & Murilo de Freitas Spinelli \\
\texttt{data/preprocessing/dataset\_builder.py} & Murilo de Freitas Spinelli, Radu Cristea \\
\hline
\multicolumn{2}{|c|}{\textit{Encoders}} \\
\hline
\texttt{data/encoders/base\_encoder.py} & Murilo de Freitas Spinelli \\
\texttt{data/encoders/event\_encoder.py} & Murilo de Freitas Spinelli \\
\texttt{data/encoders/remi\_encoder.py} & Murilo de Freitas Spinelli \\
\texttt{data/encoders/multitrack\_encoder.py} & Murilo de Freitas Spinelli \\
\hline
\multicolumn{2}{|c|}{\textit{Models}} \\
\hline
\texttt{models/base\_model.py} & Murilo de Freitas Spinelli \\
\texttt{models/lstm\_model.py} & Murilo de Freitas Spinelli \\
\texttt{models/transformer\_model.py} & Murilo de Freitas Spinelli \\
\texttt{models/model\_bundle.py} & Murilo de Freitas Spinelli \\
\hline
\multicolumn{2}{|c|}{\textit{Training}} \\
\hline
\texttt{training/trainer.py} & Murilo de Freitas Spinelli, Timofey \\
\hline
\multicolumn{2}{|c|}{\textit{Generation}} \\
\hline
\texttt{generation/base\_generator.py} & Murilo de Freitas Spinelli \\
\texttt{generation/mono\_generator.py} & Murilo de Freitas Spinelli, Ryan \\
\texttt{generation/poly\_generator.py} & Murilo de Freitas Spinelli \\
\hline
\multicolumn{2}{|c|}{\textit{CLI}} \\
\hline
\texttt{cli/experiment\_cli.py} & Murilo de Freitas Spinelli \\
\texttt{cli/generation\_cli.py} & Murilo de Freitas Spinelli \\
\texttt{cli/pipeline.py} & Murilo de Freitas Spinelli \\
\hline
\multicolumn{2}{|c|}{\textit{Tests}} \\
\hline
\texttt{tests/conftest.py} & Murilo de Freitas Spinelli \\
\texttt{tests/test\_core.py} & Murilo de Freitas Spinelli \\
\texttt{tests/test\_data\_preprocessing.py} & Murilo de Freitas Spinelli \\
\hline
\end{tabular}
\caption{Source file authorship from code documentation}
\end{table}


\section{Timeline}

The project followed an accelerated timeline after the New Year 2026 start:

\begin{itemize}
    \item \textbf{January 2026}: Project kickoff, environment setup, Sprint 1 (Monophonic Prototype)
    \item \textbf{January--February 2026}: Sprint 2 (Pseudo-Polyphonic Approach), pivot decision
    \item \textbf{February 2026}: Sprint 3 (True Polyphonic Generation), final architecture
    \item \textbf{February 2026}: Testing, documentation, report writing, final refinements
\end{itemize}

The compressed timeline demanded intensive work, particularly from the project lead, but the Scrum methodology and clear architectural vision enabled rapid progress despite the aggressive schedule.
