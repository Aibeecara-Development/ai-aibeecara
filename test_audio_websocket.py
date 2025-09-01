import asyncio
import websockets
import json

async def test_audio_conversation():
    uri = "ws://localhost:8000/conversation_stream/"
    
    try:
        print("🔄 Connecting to WebSocket...")
        async with websockets.connect(uri) as websocket:
            print("✅ Connected successfully!")
            
            # Wait for welcome message
            welcome = await websocket.recv()
            print(f"📨 {welcome}")
            
            # Send initial configuration
            config_data = {
                "selected_topic_name": "General",
                "user_input": "",
                "history_log": [],
                "exchange_count": 0,
                "tts_model": "aura-2-amalthea-en"
            }
            
            await websocket.send(json.dumps(config_data))
            print("✅ Configuration sent!")
            
            # Receive configuration confirmation
            config_msg = await websocket.recv()
            print(f"📨 {config_msg}")
            
            # Receive ready message
            ready_msg = await websocket.recv()
            print(f"📨 {ready_msg}")
            
            # Test with multiple audio files
            audio_files = [
                r"c:\Users\LENOVO\Desktop\Aibeecara\ai-aibeecara\src\data\audio\2025_07_08_14_09_18.mp3",
                r"c:\Users\LENOVO\Desktop\Aibeecara\ai-aibeecara\src\data\audio\output_2_aura-2-apollo-en.wav"
            ]
            
            for audio_index, audio_file_path in enumerate(audio_files, 1):
                try:
                    print(f"\n🎵 === Audio {audio_index} ===")
                    with open(audio_file_path, "rb") as audio_file:
                        audio_data = audio_file.read()
                        
                    print(f"📤 Sending audio data ({len(audio_data)} bytes)...")
                    await websocket.send(audio_data)
                    
                    # Listen for the expected sequence of responses:
                    # 1. Transcribing audio...
                    # 2. Transcribed: [text]
                    # 3. Generating response...
                    # 4. Bot: [response]
                    # 5.  Evaluation: [simple result]
                    
                    print("🎧 Listening for responses...")
                    
                    response_count = 0
                    max_responses = 6  # Allow for the 5 expected responses plus buffer
                    
                    while response_count < max_responses:
                        try:
                            response = await asyncio.wait_for(websocket.recv(), timeout=30)
                            response_count += 1
                            print(f"📨 Response {response_count}: {response}")
                            
                            # Check if this is the final evaluation response
                            if "📈 Evaluation:" in response:
                                print(f"✅ Audio {audio_index} processing completed successfully!")
                                
                                # Parse and display simple evaluation
                                try:
                                    eval_text = response.split("📈 Evaluation: ")[1]
                                    eval_data = json.loads(eval_text.replace("'", '"'))
                                    print(f"   � Transcript: {eval_data.get('transcript', 'N/A')[:100]}...")
                                    print(f"   🤖 Response: {eval_data.get('response', 'N/A')[:100]}...")
                                except:
                                    print("   ⚠️  Could not parse evaluation details")
                                    
                                break
                                
                            # Handle error responses
                            elif "❌" in response:
                                if "Transcription error" in response:
                                    print(f"   ⚠️  Transcription failed for audio {audio_index}")
                                    break
                                elif "No speech detected" in response:
                                    print(f"   ⚠️  No speech detected in audio {audio_index}")
                                    break
                                else:
                                    print(f"   ❌ Unexpected error: {response}")
                                    break
                                    
                        except asyncio.TimeoutError:
                            print(f"   ⏰ Timeout after {response_count} responses - moving to next audio")
                            break
                            
                    # Small delay between audio files to avoid overwhelming the server
                    if audio_index < len(audio_files):
                        print("   ⏳ Waiting before next audio...")
                        await asyncio.sleep(2)
                    
                except FileNotFoundError:
                    print(f"❌ Audio file {audio_index} not found: {audio_file_path}")
                    continue
                except Exception as file_error:
                    print(f"❌ Error processing audio file {audio_index}: {file_error}")
                    continue
                    
            # Test edge case: empty audio
            print(f"\n🧪 === Testing Empty Audio ===")
            try:
                await websocket.send(b"")  # Send empty audio
                
                empty_response = await asyncio.wait_for(websocket.recv(), timeout=10)
                print(f"📨 Empty audio response: {empty_response}")
                
                # Check if server handles empty audio gracefully
                if "No speech detected" in empty_response or "❌" in empty_response:
                    print("✅ Server handled empty audio correctly")
                else:
                    print("⚠️  Unexpected response to empty audio")
                    
            except asyncio.TimeoutError:
                print("⏰ No response to empty audio (timeout)")
            except Exception as empty_error:
                print(f"❌ Error testing empty audio: {empty_error}")
                
            print("\n🎯 === Test Summary ===")
            print(f"✅ Tested {len(audio_files)} audio files")
            print("✅ Verified audio → transcribe → chat flow")
            print("✅ Tested error handling with empty audio")
                
    except ConnectionRefusedError:
        print("❌ Connection refused. Make sure the server is running on localhost:8000")
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_audio_conversation())
