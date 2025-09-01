import asyncio
import websockets
import json

async def test_audio_conversation():
    uri = "ws://localhost:8000/conversation_stream/"
    
    async with websockets.connect(uri) as websocket:
        welcome = await websocket.recv()
        print(welcome)
        
        config_data = {
            "selected_topic_name": "General",
            "user_input": "",
            "history_log": [],
            "exchange_count": 0,
            "tts_model": "aura-2-amalthea-en"
        }
        
        await websocket.send(json.dumps(config_data))
        
        config_msg = await websocket.recv()
        print(config_msg)
        
        ready_msg = await websocket.recv()
        print(ready_msg)
        
        audio_files = [
            r"c:\Users\LENOVO\Desktop\Aibeecara\ai-aibeecara\src\data\audio\2025_07_08_14_09_18.mp3",
            r"c:\Users\LENOVO\Desktop\Aibeecara\ai-aibeecara\src\data\audio\output_2_aura-2-apollo-en.wav"
        ]
        
        for audio_file_path in audio_files:
            with open(audio_file_path, "rb") as audio_file:
                audio_data = audio_file.read()
                
            await websocket.send(audio_data)
            
            response_count = 0
            max_responses = 5
            
            while response_count < max_responses:
                response = await asyncio.wait_for(websocket.recv(), timeout=30)
                response_count += 1
                print(response)
            
            await asyncio.sleep(2)
        
        await websocket.send(b"")
        empty_response = await asyncio.wait_for(websocket.recv(), timeout=10)
        print(empty_response)

if __name__ == "__main__":
    asyncio.run(test_audio_conversation())
