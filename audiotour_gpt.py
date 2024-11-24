import streamlit as st
import whisper
import numpy as np
import torch
from pydub import AudioSegment
import openai


# Load the Whisper model
model = whisper.load_model("base")  # You can change this to "tiny", "small", "medium", or "large"

def load_audio_file(audio_file):
    """Load audio file into a format that Whisper can process."""
    audio = AudioSegment.from_file(audio_file)
    audio = audio.set_channels(1).set_frame_rate(16000)  # Whisper works better with 1 channel (mono) and 16kHz sample rate
    
    # Convert the audio to a numpy array and normalize to [-1, 1]
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)  # Convert to float32
    samples = samples / np.max(np.abs(samples))  # Normalize to [-1, 1]
    
    # Convert to a torch tensor
    audio_tensor = torch.from_numpy(samples)
    return audio_tensor

def transcribe_audio(audio):
    """Transcribe audio using Whisper."""
    result = model.transcribe(audio)
    return result['text']

def get_research_topics(result):
    """Extracts research topics from user ideas."""
    try:
        prompt = (
            "Given the following input, extract the key topics, terms, and locations that require additional research. \n\n"
            "Generate search queries to gather relevant information for these topics. Ensure the queries are concise and focused, covering historical context, significance, or interesting facts.\n\n"

            "Input:\n\n"
            f"{result}\n\n"

        )
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3000,
            temperature=0.3
        )
        return response.choices[0].message['content']
    except Exception as e:
        print(f"Error during topic extraction: {e}")
        return None


def fact_check_information(research_topics):
    """Performs research and fact-checking based on topics."""
    try:
        prompt = (
            "You are a fact-checking expert. Using the following topics or queries, provide accurate and detailed "
            "information for each, ensuring all facts are verified:\n\n"
            f"{research_topics}\n\n"
            "Output the results in a structured and detailed format."
        )
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=3000,
            temperature=0.3
        )
        return response.choices[0].message['content']
    except Exception as e:
        print(f"Error during fact-checking: {e}")
        return None


def create_outline(result, research_results):
    """Combines user ideas and research results into a detailed outline."""
    try:
        prompt = (
            "Create a detailed outline for an audio tour script using the following:\n\n"
            f"User Ideas:\n{result}\n\n"
            f"Research Results:\n{research_results}\n\n"
            "Structure the outline as:\n"
            "1. Introduction: Briefly summarize the tour's theme.\n"
            "2. Main Sections: Divide content into sections. For each, include a title, key points, and transitions.\n"
            "3. Conclusion: Summarize closing remarks.\n\n"
            "Output the outline in a clear, organized format."
        )
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=16000,
            temperature=0
        )
        return response.choices[0].message['content']
    except Exception as e:
        print(f"Error during outline creation: {e}")
        return None


def generate_script(outline):
    """Generates a complete audio tour script from the outline."""
    try:
        prompt = (
            "Using the following outline, craft a complete audio tour script:\n\n"
            f"{outline}\n\n"
            "The script should include engaging language, natural transitions, and sensory descriptions. Use cues like "
            "'Now, let's move to...' or 'Imagine...' to guide the listener."
        )
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=16000,
            temperature=0
        )
        return response.choices[0].message['content']
    except Exception as e:
        print(f"Error during script generation: {e}")
        return None


# Streamlit UI
st.title("CineWav Audio Tour Script Generator ")
st.write("Upload an audio file to get transcribed text, followed by research and script generation.")

# File uploader
audio_file = st.file_uploader("Upload an audio file", type=["mp3", "m4a", "wav", "flac"])

if audio_file is not None:
    st.audio(audio_file, format="audio/wav")  # Preview the uploaded audio
    st.write("Processing the file...")

    try:
        # Load and process audio file
        audio = load_audio_file(audio_file)
        
        # Step 1: Transcribe the audio
        transcription = transcribe_audio(audio)

        # Display transcription
        st.subheader("Transcription Result")
        st.write(transcription)

        # Step 2: Extract research topics from the transcription
        st.write("Extracting research topics...")
        research_topics = get_research_topics(transcription)
        if research_topics:
            st.subheader("Research Topics")
            st.write(research_topics)

        # Step 3: Perform fact-checking based on research topics
        st.write("Fact-checking information...")
        research_results = fact_check_information(research_topics)
        if research_results:
            st.subheader("Fact-Checked Information")
            st.write(research_results)

        # Step 4: Create an outline based on the research results
        st.write("Creating outline...")
        outline = create_outline(transcription, research_results)
        if outline:
            st.subheader("Outline")
            st.write(outline)

        # Step 5: Generate the audio tour script based on the outline
        st.write("Generating script...")
        script = generate_script(outline)
        if script:
            st.subheader("Generated Script")
            st.write(script)

    except Exception as e:
        st.error(f"Error processing the audio: {e}")
