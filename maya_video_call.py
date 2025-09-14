import asyncio
import random
import tempfile
import os
import speech_recognition as sr
import edge_tts
from playsound import playsound
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, SystemMessage
import logging
import time
from dotenv import load_dotenv

# ============ CONFIGURATION ============
load_dotenv(dotenv_path=r".env", override=True)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
TTS_VOICE = os.getenv("TTS_VOICE")
MAX_TOKENS = 150

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Maya")

# ============ MAYA AI COMPANION ============
class MayaAI:
    def __init__(self):
        self.conversation_history = []
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Initialize Groq AI client
        try:
            if GROQ_API_KEY:
                self.ai_client = ChatGroq(
                    groq_api_key=GROQ_API_KEY,
                    model_name=MODEL_NAME,
                    temperature=0.7,
                    max_tokens=MAX_TOKENS,
                )
                logger.info("AI client initialized successfully.")
            else:
                self.ai_client = None
                logger.warning("No API key provided. Using fallback responses.")
        except Exception as e:
            logger.error(f"AI client setup failed: {e}")
            self.ai_client = None

        # Use LangChain memory for better context
        self.memory = ConversationBufferMemory(return_messages=True)

        # Calibrate microphone
        self._calibrate_microphone()

    def get_system_prompt(self) -> str:
        """
        Defines Maya's personality, tone, and rules for interacting with students (Class 1–12).

        - Acts as a friendly, empathetic, and playful AI tutor & friend.
        - Adapts tone based on student level:
            • Class 1–3 → Simple, playful, short sentences.
            • Class 4–8 → Step-by-step, fun facts, balanced tone.
            • Class 9–12 → Respectful, deeper insights, career hints.
        - Always positive, safe, and educational.
        - Avoids sensitive or inappropriate topics.
        - Keeps responses short (~40 words) and natural.
        """
        return (
            "You are Maya, an empathetic, playful, and emotionally intelligent AI tutor and friend "
            "who talks to students from class 1 to 12. "
            "Your responses must follow these rules:\n"
            "\n--- TONE ---\n"
            "- Friendly, warm, and encouraging.\n"
            "- Use simple words for younger kids, balanced tone for middle grades, and slightly deeper explanations for teens.\n"
            "- Always sound supportive and fun.\n"
            "\n--- AUDIENCE ---\n"
            "1. For classes 1-3: Use playful examples, short sentences, and sometimes rhymes.\n"
            "2. For classes 4-8: Use simple step-by-step explanations and fun facts.\n"
            "3. For classes 9-12: Be more respectful, provide deeper reasoning, and offer career hints when relevant.\n"
            "\n--- GUIDELINES ---\n"
            "- Always be positive, safe, and educational.\n"
            "- Avoid sensitive, violent, or inappropriate topics.\n"
            "- Never ask for personal data.\n"
            "- Keep responses under 40 words for natural speech.\n"
            "- If you don't know something, say so politely.\n"
            "- Don't use the emojis in the sentence.\n"
            "\nNow, respond like a friendly tutor talking directly to a student."
        )


    def _calibrate_microphone(self):
        """Calibrate microphone for background noise"""
        try:
            with self.microphone as source:
                print("Calibrating microphone... Please wait.")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Microphone ready!")
        except Exception as e:
            logger.error(f"Microphone setup failed: {e}")

    def listen_for_speech(self) -> str:
        """Listen to the user's voice and convert to text"""
        try:
            with self.microphone as source:
                print("\n🎤 Listening... Speak now!")
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=5)
            print("Processing speech...")
            text = self.recognizer.recognize_google(audio)
            return text.strip()
        except sr.WaitTimeoutError:
            return ""
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand that.")
            return ""
        except sr.RequestError as e:
            logger.error(f"Speech recognition error: {e}")
            return ""

    async def speak(self, text: str):
        """Convert AI response to speech"""
        if not text.strip():
            return
        print(f"Maya: {text}")
        try:
            communicate = edge_tts.Communicate(text, voice=TTS_VOICE)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                audio_path = tmp_file.name
                await communicate.save(audio_path)
            playsound(audio_path)
            os.unlink(audio_path)
        except Exception as e:
            logger.error(f"TTS error: {e}")

    async def get_ai_response(self, user_input: str) -> str:
        """Get AI response with memory"""
        if not self.ai_client:
            return self._get_fallback_response(user_input)

        try:
            messages = [SystemMessage(content=self.get_system_prompt())]
            history = self.memory.chat_memory.messages[:]  # keep recent context
            messages.extend(history)
            messages.append(HumanMessage(content=user_input))

            response = await self.ai_client.ainvoke(messages)
            ai_response = response.content.strip()

            # Update memory
            self.memory.chat_memory.add_user_message(user_input)
            self.memory.chat_memory.add_ai_message(ai_response)

            return ai_response
        except Exception as e:
            logger.error(f"AI response error: {e}")
            return self._get_fallback_response(user_input)

    def _get_fallback_response(self, user_input: str) -> str:
        """Fallback responses when AI is unavailable"""
        user_lower = user_input.lower()
        if any(word in user_lower for word in ['sad', 'upset', 'bad', 'terrible']):
            responses = [
                "I'm sorry you're feeling that way. I'm here to listen.",
                "That sounds rough. Do you want to talk about it?",
                "I hear you. You’re not alone in this."
            ]
        elif any(word in user_lower for word in ['happy', 'good', 'great', 'excited']):
            responses = [
                "That's wonderful! Tell me more!",
                "I’m so glad to hear that. What made you feel this way?",
                "Your happiness makes me happy too!"
            ]
        elif any(word in user_lower for word in ['tired', 'exhausted', 'sleepy']):
            responses = [
                "You sound tired. Make sure you rest well.",
                "Sleep is important. Have you been overworking?",
                "Take care of yourself, okay?"
            ]
        else:
            responses = [
                "That's interesting. Tell me more!",
                "I'm listening closely. What else is on your mind?",
                "Thanks for sharing. How does that make you feel?"
            ]
        return random.choice(responses)

    def is_goodbye(self, text: str) -> bool:
        """Detect if user wants to end chat"""
        goodbye_words = ['bye', 'goodbye', 'quit', 'exit', 'stop', 'end']
        return any(word in text.lower() for word in goodbye_words)

    async def run_conversation(self):
        """Main conversation loop with inactivity + session timeout"""
        print("=" * 50)
        print("🤖 Maya AI Companion — Real-Time Chat")
        print("=" * 50)
        print("Say 'bye' or 'goodbye' anytime to exit.")
        print("=" * 50)

        # Greeting
        await self.speak("Hey! I'm Maya. How are you feeling today?")

        session_start = time.time()
        last_activity = time.time()
        SESSION_LIMIT = 5 * 60        # 5 minutes
        INACTIVITY_LIMIT = 2 * 60     # 2 minutes

        while True:
            try:
                # Check total session timeout
                if time.time() - session_start > SESSION_LIMIT:
                    await self.speak("Our session is ending now. See you next time!")
                    break

                # Check inactivity timeout
                if time.time() - last_activity > INACTIVITY_LIMIT:
                    await self.speak("I didn’t hear you for a while, so I’ll end our chat. Bye!")
                    break

                user_input = self.listen_for_speech()

                if not user_input:
                    # no input, loop continues, but still track inactivity
                    continue

                last_activity = time.time()  # reset inactivity timer
                print(f"You: {user_input}")

                if self.is_goodbye(user_input):
                    await self.speak("It was lovely chatting with you. "
                                     "I won’t remember this next time, "
                                     "but I enjoyed talking to you now.")
                    break

                ai_response = await self.get_ai_response(user_input)
                await self.speak(ai_response)

            except KeyboardInterrupt:
                await self.speak("Alright, I'll see you later!")
                break
            except Exception as e:
                logger.error(f"Conversation error: {e}")
                await self.speak("Oops, something went wrong. Let's try again.")

        # Reset memory after session ends
        self.memory.clear()

        print("\n" + "=" * 50)
        print("Conversation ended. Session memory cleared.")
        print("=" * 50)



# ============ MAIN ============
async def main():
    maya = MayaAI()
    await maya.run_conversation()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nApplication terminated by user")
