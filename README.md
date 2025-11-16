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
 -->
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

### 3. `/transcribe/` (POST)
**Description:**
Returns transcription of audio URL using Deepgram ASR.

**Input:**
```json
{
  "audio_url": "<audio_url>"
}
```

**Output:**
```json
{
   "response": {...},
  "transcript": "This is the original transcribed text.",
   "waveform": "....."
}
```

### 4. `/evaluate/` (POST)

**Description:**
Evaluates current audio (ACCUMULATE IN FRONT-END SO THAT IT CAN BE AVERAGED IN THE END).

### The response from `/transcribe/` endpoint acts as the input to this endpoint.

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
            "transcript": "This is the original transcribed text.",
            "corrected_transcript": "This is the corrected transcribed text.",
            "grammar_score": 0.97,
            "grammar_explanation": "The verb 'go' does not agree with the subject 'he.' It should be 'goes.'",
            "tense_used": "Present simple tense",
            "pause_score_dict": {
                 "score": 0.88,
                 "pause_between_words": [{'start_word': 'and', 'end_word': 'we', 'duration': 2.5600001000000003}, 
                    {'start_word': 'that', 'end_word': 'to', 'duration': 1.8399999999999999}, ...],
                 "pause_transcript": "I, uh, and… we…"
             },
            "stutter_score": 0.88,
            "stuttered_phrases": ['uh', 'um', 'you know', 'like'],
            "fluency_score": 0.88,
            "fluency_speed": "Fluent",
            "speech_rate": {
                 "wpm": 67,
                 "spm": 76
             },
            "pronunciation_score":  {
             "score": 97.0,
             "words": [
                 {
                     "Real words": "In",
                     "Transcribed words": "In",
                     "Highlights": "*I**n*",
                     "Predicted phonemes": "ɪn",
                     "Ground truth phonemes": "ɪn",
                     "Pronunciation result": true
                 },
                 {
                     "Real words": "the",
                     "Transcribed words": "the",
                     "Highlights": "*t**h**e*",
                     "Predicted phonemes": "ðə",
                     "Ground truth phonemes": "ðə",
                     "Pronunciation result": true
                 },
            "vocabulary_score":
               {
                 "statistics": {
                   "A1": 19,
                   "A2": 4,
                   "B1": 7,
                   "B2": 1,
                   "C1": 0,
                   "C2": 1
                 },
                 "tokens": [
                   {
                     "word": "in",
                     "lemma": "in",
                     "pos": "IN",
                     "level_score": 1.0,
                     "cefr": "A1", 
                      "pronunciation": "dəmeɪn",
                      "definition": "people in general; especially a distinctive group of people with some shared interest",
                      "example_sentence": "The forest was full of wildlife."

                   },
                   {
                     "word": "world",
                     "lemma": "world",
                     "pos": "NN",
                     "level_score": 1.0,
                     "cefr": "A1",
                      "pronunciation": "dəmeɪn",
                      "definition": "people in general; especially a distinctive group of people with some shared interest",
                      "example_sentence": "The forest was full of wildlife.",

                     "synonyms": [
                         {
                             "synonym": "domain",
                             "pos": "NN",
                             "pronunciation": "dəmeɪn",
                             "definition": "people in general; especially a distinctive group of people with some shared interest",
                             "level_score": 2.72,
                             "cefr": "B1",
                             "example_sentence": "the western domain"
                         },
                         {
                             "synonym": "reality",
                             "pos": "NN",
                             "pronunciation": "ɹɪælᵻɾi",
                             "definition": "all of your experiences that determine how things appear to you",
                             "level_score": 3.0,
                             "cefr": "B1",
                             "example_sentence": "his reality was shattered"
                         },
                   ...
                 ]
               }
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
Before using this, make sure that evaluate_cefr_stats, evaluate_pronunciation, evaluate_pause, and
evaluate_stutter functions are integrated into the chatbot API so that it can be evaluated every time the user
makes an input. In the end, grab the scores from each bubble chat from the user, put them into array, and send the
request so that the evaluation results are averaged and returned to the user.

**Input:**

```json
{
  "history_log": [
    ["user", "I go school everyday."],
    ["bot", "Good try!"]
  ],
  "exchange_count": 1,
  "pronunciation_array": [0.8, 0.6, 0.7],
  "pause_array": [0.8, 0.9, 0.7],
  "stutter_array": [0.7, 0.6, 0.8]
}
```

**Output:**

```json
{
  "original_message": "An original message with bad grammar.",
  "corrected_transcript": "A corrected message with good grammar.",
  "grammar_score": 0.6,
  "vocabulary_score": 0.7,
  "pronunciation_score": 0.7,
  "total_score": 0.7
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
  "history_log": [
    ["user", "I wake up early"],
    ["bot", "Good job! What do you do after?"]
  ],
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

### 11. `/try_by_yourself/` (POST)
**Description:**
Updating the score of a certain aspect (grammar, vocabulary, pronunciation, pause, stutter) by replying with an audio.

**Input:**

```json
{
  "corrections": [
    {"chat_bubble_id": 1, "score": 0.75},
    {"chat_bubble_id": 2, "score": 0.85},
    {"chat_bubble_id": 3, "score": 0.90}
  ],
  "aspect_score": {
    "grammar_score": 0.80,
    "vocabulary_score": 0.85,
    "pronunciation_score": 0.70,
    "fluency_score": 0.90
  },
  "aspect": "pronunciation",
  "audio_url": "https://example.com/audio/sample.wav",
  "chat_bubble_correction_id": 2,
   "correction_text": "corrected text"
}
```

**Output:**

```json
{
   "new_score": 0.8, 
   "new_aspect_mean": 0.79, 
   "new_total_score": 0.89,
   "new_speed": "Fluent"
}
```

### 12. `/conversation_stream/` (WebSocket)

**Description:**
Streams conversation history and AI responses in real-time.

**Input:**

```json
{
  "selected_topic_name": "Daily Routine",
  "user_input": "I usually wake up at 7am and brush my teeth.",
  "history_log": [
    ["user", "I wake up early"],
    ["bot", "Good job! What do you do after?"]
  ],
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
   uvicorn api:app --reload
   ```

# TTS Options Available:

Default TTS model: `aura-2-amalthea-en` (american feminine)
Accent TTS choices: `american`, `british`, `australian`
Gender TTS choices: `masculine`, `feminine`

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

# evaluate_stutter functions into this websocket API so that it can be evaluated every time the user

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
