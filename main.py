import os
import requests
from bs4 import BeautifulSoup
import re
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from scipy.io.wavfile import write
from IPython.display import Audio

# Run from the directory containing BLIP and XTTS directories



# Fetching text and images from a URL
def fetch_text_and_images(url):
    response = requests.get(url)
    web_content = response.content
    
    soup = BeautifulSoup(web_content, 'html.parser')
    
    text_content = soup.get_text(separator=' ', strip=True)
    clean_text = re.sub(r'\s+', ' ', text_content)
    print("Text scraped from", url)
    print(clean_text)
    
    image_urls = []
    print("\nImages scraped from", url)
    for item in soup.find_all('img'):
        if 'src' in item.attrs:
            img_url = item['src']
            if img_url.startswith('http://') or img_url.startswith('https://'):
                image_urls.append(img_url)
                print(img_url)
    
    return clean_text, image_urls

# Generating captions for images using the BLIP model
def generate_image_captions(image_urls):
    # Loading the BLIP processor and model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    captions = []
    
    for img_url in image_urls:
        try:
            # Download the image from the URL
            response = requests.get(img_url)
            raw_image = Image.open(BytesIO(response.content)).convert('RGB')
            
            # Conditional image captioning
            text = "a picture of "
            inputs = processor(raw_image, text, return_tensors="pt")
            
            out = model.generate(**inputs, max_new_tokens=1000)
            caption = processor.decode(out[0], skip_special_tokens=True)
            captions.append(caption)
        except UnidentifiedImageError:
            # Skip the image if it cannot be identified
            print(f"Skipping unidentifiable image: {img_url}")
        except Exception as e:
            # General exception handling for other errors
            print(f"An error occurred with image {img_url}: {e}")

    print(captions)    
    return captions

# Converting text to speech using XTTS
def text_to_speech(text):
    config = XttsConfig()
    config.load_json("./XTTS-v2/config.json")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir="./XTTS-v2/")
    
    reference_audios = ["./sample_voice1.wav", "./sample_voice2.wav"]
    
    # Helper function to split text into chunks of 225 characters
    def split_text(text, max_length=225):
        return [text[i:i + max_length] for i in range(0, len(text), max_length)]
    
    text_chunks = split_text(text)
    
    if not os.path.exists('./outputs'):
        os.makedirs('./outputs')
    
    for i, chunk in enumerate(text_chunks):
        outputs = model.synthesize(
            chunk,
            config,
            speaker_wav=reference_audios,
            gpt_cond_len=3,
            language="en",
        )
        
        audio_data = outputs['wav']
        output_file_path = f'./outputs/output_audio_{i + 1}.wav'
        write(output_file_path, 24000, audio_data)
        print(f"Audio segment {i + 1} saved at:", output_file_path)

# Main function to integrate all functionalities
def main(url):
    # Fetch text and images from the URL
    text, image_urls = fetch_text_and_images(url)
    
    # Generate captions for the images
    captions = generate_image_captions(image_urls)
    
    # Combine text and captions into a single text
    combined_text = text + " " + " ".join(captions)
    
    # Convert the combined text to speech
    text_to_speech(combined_text)

# URL to scrape
url = 'https://www.flatsound.org/'

# Run the main function
main(url)