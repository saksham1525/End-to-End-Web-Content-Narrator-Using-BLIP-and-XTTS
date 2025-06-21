Web Scraper + Image Captioning + Text-to-Speech (BLIP + XTTS)

This project scrapes a webpage for text and images, uses a vision-language model (BLIP) to generate captions for the images, and then converts the combined text into speech using the XTTS-v2 model.

Features:
- Scrapes text and images from a URL
- Generates captions for images using the BLIP model
- Synthesizes speech using XTTS-v2 with reference voice samples
- Saves the output audio files locally

Setup Instructions:

1. Clone the repository and navigate into it.

2. Install dependencies:
   pip install -r requirements.txt

Sample Reference Audios:

Sample audio files used for voice reference are located in the directory:
- sample_voice1.wav
- sample_voice2.wav

You may replace them with your own .wav files. Make sure to update the paths in the code accordingly.

3. XTTS-v2 Setup:

The project uses XTTS-v2 from Coqui TTS. Due to size and licensing restrictions, model checkpoint files and config.json are not included in the repository.

To use XTTS-v2:
- Download model files and config.json from the official Coqui GitHub or Hugging Face.
- Place them in a folder named 'XTTS-v2/' in the project root as shown:

project-root/
├── XTTS-v2/
│   ├── config.json
│   └── [model checkpoint files]

4. BLIP Model Setup:

BLIP is automatically downloaded via HuggingFace's transformers library the first time the code runs. No manual setup is required.

Running the Script:

Open the main.py file and update the URL as needed:
url = 'https://www.example.com'
Then run:
python main.py

Output audio files will be saved in the 'outputs/' directory.

Additional Notes:

- Ensure you run the script from the root directory where main.py is located.
- The script assumes XTTS-v2 and sample_audio directories are also in the root.
