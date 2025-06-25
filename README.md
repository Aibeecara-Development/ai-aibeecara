# Speech Grammar Pipeline

This project is designed to process audio files by transcribing them using the Deepgram API and then correcting the grammar of the transcriptions using the Gemini model. The pipeline consists of several modules that handle audio processing, grammar correction, and orchestration of the entire workflow.

## Project Structure

```
speech-grammar-pipeline
├── src
│   ├── audio_processing
│   │   ├── deepgram_transcriber.py  # Functions for audio transcription using Deepgram API
│   │   └── __init__.py               # Marks the directory as a Python package
│   ├── grammar_correction
│   │   ├── gemini_corrector.py       # Functions for grammar correction using Gemini model
│   │   └── __init__.py               # Marks the directory as a Python package
│   ├── pipeline.py                    # Main entry point for the pipeline
│   └── types
│       └── index.py                   # Defines custom types or interfaces
├── data
│   ├── audio                          # Directory for storing audio files
│   └── transcript_ref                 # Directory for storing reference transcripts
├── requirements.txt                   # Lists project dependencies
└── README.md                          # Project documentation
```

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd speech-grammar-pipeline
   ```

2. **Install dependencies**:
   It is recommended to use a virtual environment. You can create one using `venv` or `conda`.
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Ensure that you have the necessary API keys for Deepgram and Gemini. You can create a `.env` file in the root directory with the following content:
   ```
   DEEPGRAM_KEY=<your_deepgram_api_key>
   GEMINI_KEY=<your_gemini_api_key>
   ```

## Usage

To run the pipeline, execute the following command:
```
python src/pipeline.py
```

This will process all audio files located in the `data/audio` directory, transcribe them, and then apply grammar corrections to the transcriptions. The corrected transcripts will be available for review.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.