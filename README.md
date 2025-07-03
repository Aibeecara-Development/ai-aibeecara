# Aibeecara API Model

## Setup Instructions

1. **Clone the repository**

2. **Install dependencies using requirements.txt**

3. **Set up environment variables**:
   Ensure that you have the necessary API keys for Deepgram and Gemini. You can create a `.env` file in the `src` 
   directory with the following content:
   ```
   DEEPGRAM_KEY=<your_deepgram_api_key>
   GEMINI_KEY=<your_gemini_api_key>
   ```
   As well as `.env` files with `GEMINI_KEY` and `DEEPGRAM_KEY` in `evaluation/grammar-correction` and `evaluation/speech-recognition` respectively if needed.

## Usage

To run the pipeline, execute the following command:
```
python src/pipeline.py
```

## Logs

### 29 Jun 2025 
- **Pipeline**: Pipeline execution (ASR + grammar correction) completed successfully
- **Deepgram**: Uses Deepgram model for ASR.
- **Gemini**: Uses Gemini model for grammar correction.
