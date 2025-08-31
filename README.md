# Aibeecara API Model

## Setup Instructions

1. **Clone the repository**

2. **Install dependencies using requirements.txt**

3. Download the LLAMA 2 model here: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_K_S.gguf
4. Place the downloaded model in the `src/data/` directory.

5. **Set up environment variables**:
   Ensure that you have the necessary API keys for Deepgram and Gemini. You can create a `.env` file in the `src` 
   directory with the following content:
   ```
   DEEPGRAM_KEY=<your_deepgram_api_key>
   GEMINI_KEY=<your_gemini_api_key>
   ```
   As well as `.env` files with `GEMINI_KEY` and `DEEPGRAM_KEY` in `evaluation/grammar-correction` and `evaluation/speech-recognition` respectively if needed.

# IMPORTANT: 
Make sure to increment the exchange_count in the request body for each new user input in the chatbot.

## API Endpoints

<!--
### 1. `/chat/` (POST)

**Description:**  
Stream AI-generated response based on selected roleplay topic and conversation history.

**Input:**
```json
{
  "selected_topic_name": "Daily Routine",
  "user_input": "I usually wake up at 7am and brush my teeth.",
  "history_log": [["user", "I wake up early"], ["bot", "Good job! What do you do after?"]],
  "exchange_count": 1,
  "tts_model": "aura-2-thalia-en"
}
```

**Output:**
```json
{
  "response": "That's a great start to your day! After brushing your teeth, what do you usually do next?"
}
```

### 2. `/ws/chat/` (WebSocket)

**Description:**  
Real-time streaming chat endpoint using WebSocket.

**Input:**
```json
{
  "selected_topic_name": "Travel",
  "user_input": "I went to Japan last year.",
  "history_log": [["user", "Have you ever been abroad?"], ["bot", "Yes!"]],
  "exchange_count": 2,
  "tts_model": "aura-2-apollo-en"
}
```

**Output:**
Real-time AI response streamed as `send_text()` chunks

### 3. `/ws/chat/summary` (WebSocket)

**Description:**  
Generates a summary and grammar feedback after a conversation.

**Input:**
```json
{
  "history_log": [
    ["user", "I go to school everyday."],
    ["bot", "That's great! What do you study?"],
    ["user", "I study math and science."]
  ],
  "user_input": "I like science because it's interesting."
}
```

**Output:**
Textual summary and feedback sent as WebSocket text messages.

### 4. `/transcribe/` (POST)

**Description:**  
Transcribes uploaded audio and evaluates for pauses and repetition.

**Input:**
Form file upload (multipart/form-data) with key: file.
```json
{
  "audio_url": "<audio_url>"
}
```

**Output:**
```json
{
  "transcript": "I like to read books. I like to read books.",
  "pause_score": 0.85,
  "pause_details": [
    {
      "start_word": "books",
      "end_word": "I",
      "duration": 3.5
    }
  ],
  "repetition_score": 0.75,
  "repeated_phrases": ["i like to read books"]
}
```

### 5. `/chat/tts/` (POST)

**Description:**  
Generates a TTS `.wav` file from input text using Deepgram TTS with adjustable voice and speed.

**Input:**
```json
{
  "text": "Hello, how are you today?",
  "accent": "american",
  "gender": "feminine",
  "speed": 1.0
}
```


**Output:**
Returns `.wav` audio file response (`audio/wav`).
 -->

### 6. `/chat/topic/` (POST)
**Description:**
Validates whether or not a topic is valid (BROAD or NARROW).

**Input:**
```json
{
  "selected_topic_name": "Daily Routine"
}
```

**Output:**
```json
{
  "validation": "BROAD"
}
```

#### 7. `/chat/emotion/` (POST)
**Description:**
Detects emotion from what the chatbot says and returns the detected emotion.

**Input:**
```json
{
  "model_output": "I am feeling very happy today!"
}
```

**Output:**
```json
{
  "emotion": "happy"
}
```

### 8. `/evaluate/` (POST)

**Description:**  
Performs grammar evaluation of user's conversation based on chat history.

**Input:**
```json
{
  "history_log": [
    ["user", "I go school everyday."],
    ["bot", "Good try!"]
  ],
  "exchange_count": 1
}
```

**Output:**
```json
{
  "original_message": "user's message here",
   "corrected_transcript": "corrected message here",
   "evaluation_score": 0.78,
   "vocabulary_score": {
     "statistics": {
       "A1": 69,
       "A2": 34,
       "B1": 23,
       "B2": 11,
       "C1": 2,
       "C2": 7
     },
     "tokens": [
       {
         "word": "going",
         "lemma": "go",
         "pos": "VERB",
         "level_score": 1.23,
         "cefr": "A1"
       },
       {
         "word": "university",
         "lemma": "university",
         "pos": "NOUN",
         "level_score": 3.41,
         "cefr": "B1"
       },
        ...
     ]
}
}
```

### 9. `/translate/` (POST)
**Description:**
Translates user input text to Indonesian using Google Translate API.

**Input:**
```json
{
  "selected_topic_name": "Daily Routine",
  "user_input": "I usually wake up at 7am and brush my teeth.",
  "history_log": [["user", "I wake up early"], ["bot", "Good job! What do you do after?"]],
  "exchange_count": 1,
  "tts_model": "aura-2-thalia-en"
}
```

**Output:**
```json
{
  "translated_text": "Saya biasanya bangun jam 7 pagi dan menyikat gigi."
}
```

### 10. `/hint/` (POST)
**Description:**
Provides hints for the user based on the last chatbot output.

**Input:**
```json
{
  "response": "Do you like pizza with pineapple?"
}
```

**Output:**
```json
{
  "hint": "I thought it was meh."
}
```

### 11. `/conversation_stream/` (WebSocket)
**Description:**
Streams conversation history and AI responses in real-time.

**Input:**
```json
{
  "selected_topic_name": "Daily Routine",
  "user_input": "I usually wake up at 7am and brush my teeth.",
  "history_log": [["user", "I wake up early"], ["bot", "Good job! What do you do after?"]],
  "exchange_count": 1,
  "tts_model": "aura-2-thalia-en"
}
```

**Output:**
```json
{"type": "transcript", "text": "Hello, I want to travel."},
{"type": "chat", "text": "Great! Where do you want to go?"},
{"type": "tts", "audio": "<base64...>"},
{"type": "tts", "audio": "<base64...>"},
{"type": "tts", "audio": "<base64...>"},
```

## Setup & Run Instructions

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Set your API keys in the `.env` file as described above.**
   ```bash
   GEMINI_KEY=your_google_gemini_api_key
   DEEPGRAM_API_KEY=your_deepgram_api_key
   ```
3. Run the FastAPI server:
   ```bash
   cd src
   uvicorn src.api:app --reload
   ```

## Topics Available
- General 
- School Subjects 
- At Home Activities 
- Hobbies
- Lifestyle
- Friendship and Achievements
- Global Issues
- Custom User Topic

## TO-DO:
- # Don't forget to integrate the evaluate_transcription, evaluate_cefr_stats, evaluate_pronunciation, evaluate_pause, and
# evaluate_repetition functions into this websocket API so that it can be evaluated every time the user
# makes an input. In the end, the evaluation results are averaged and returned to the user.

- # Don't forget to set API functions like translate_text, chat_topic, chat_emotion, etc into async or non-async 
# functions.

## Logs

### 29 Jun 2025 
- **Pipeline**: Pipeline execution (ASR + grammar correction) completed successfully
- **Deepgram**: Uses Deepgram model for ASR.
- **Gemini**: Uses Gemini model for grammar correction.

### 2 Jul 2025
- **Add whisper model to timeline**: Added Whisper model to the pipeline for ASR.

### 3 Jul 2025
- **Add chatbot simulation**: Added Gemini chatbot simulation.
- **Dialogue template**: Created a dialogue template for the chatbot simulation.

### 4 Jul 2025
- **Add streaming chatbot**: Implemented a streaming chatbot using Gemini.

### 7 Jul 2025
- **Add summarization**: Added summarization of the conversation to the pipeline.

### 12 Jul 2025
- **Add text-to-speech**: Implemented text-to-speech functionality using Deepgram.

### 22 Jul 2025
- **Add FastAPI**: Integrated FastAPI for serving the pipeline as a web service.
- **Add pause scoring in voice activity detection**: Implemented pause scoring in voice activity detection to improve accuracy.
- **Add grammar correction**: Added grammar correction functionality to the pipeline using Huggingface model.

### 1 Aug 2025
- **Add TTS options**: Added extra options for text-to-speech (TTS) in the API such as gender, speed, and accent selection.
- **Add conversation topics**: Implemented conversation topic prompts to enhance the chatbot's context awareness.

### 3 Aug 2025
- **Add evaluation endpoint**: Added an endpoint for evaluating grammar and vocabulary based on conversation history.
- **Add websocket support**: Implemented WebSocket support for real-time chat interactions.

### 8 Aug 2025
- **Add vocabulary scoring**: Added vocabulary scoring to the evaluation endpoint to assess language proficiency.
- **Add vocabulary endpoint**: Created a dedicated endpoint for vocabulary evaluation based on user input.

### 10 Aug 2025
- **Change TTS model**: Switched TTS model to aura-2-amalthea-en for better performance.

### 13 Aug 2025
- **Add emotion detection**: Implemented emotion detection based on chatbot responses to enhance user interaction.
- **Add topic validation**: Added an endpoint to validate conversation topics as either broad or narrow.