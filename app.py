import gradio as gr
import google.generativeai as genai
#from google.colab.patches import cv2_imshow
import cv2
from PIL import Image
import math
import os
from gtts import gTTS
from playsound import playsound
from PIL import PngImagePlugin
import time

gemini_API = os.environ["gemini_API"] = "AIzaSyA6NI9yl3J-njCCFurFu6VGWQ8zEycK-HY"
genai.configure(api_key=gemini_API)

for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)

vision_model = genai.GenerativeModel('gemini-pro-vision')
text_model = genai.GenerativeModel('gemini-pro')

def caption_generation(image):
  prompt = "Explain what is happening in the image."
  response = vision_model.generate_content([f"{prompt}", image], stream=True)
  response.resolve()
  gen_caption = response.text
  return gen_caption

def split_frames(video):

  cap = cv2.VideoCapture(video)

  local_captions = []

  fps = cap.get(cv2.CAP_PROP_FPS)

  print(f"FPS : {fps}")

  #interval = math.ceil(fps/2)

  #print(f"interval : {interval}")

  frame_interval = int(2 * fps)

  frame_count = 0
  extracted_frames_count = 0
  extracted_frames = []

  while True:
      ret, frame = cap.read()
      if not ret:
          break

      if frame_count % frame_interval == 0:
          #cv2.imshow("extracted frame", frame)
          frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          img = Image.fromarray(frame_rgb)
          gen_caption = caption_generation(img)
          local_captions.append(gen_caption)
          extracted_frames.append(frame_rgb)

          extracted_frames_count += 1
      frame_count += 1

  #print(local_captions)
  #print(f"extracted frames : {extracted_frames}")
  return local_captions, extracted_frames_count, extracted_frames

def condensation(local_captions):
  prompt = "Explain the scenario of what is happening based on the input captions given like a brief summary. Combine all the captions generated from images and summarize them"
  response = text_model.generate_content(f"{prompt}. {local_captions}", stream=True)
  response.resolve()
  return response.text

def video_understanding(video):
  local_captions = split_frames(video)
  summary = condensation(local_captions)
  print(summary)
  text_to_audio(summary)

def text_to_audio(summary):
    tts = gTTS(text=summary, lang='en')
    filename = 'output.mp3'
    tts.save(filename)
    time.sleep(3)
    return filename
    #print('playing sound')
    #playsound(filename)
    #os.close(filename)

interface_description="""<p>V1.3: Changed Caption generation layer with Gemini vision pro architecture. Can perform Video understanding and can handle short videos upto 15 seconds.,</p>
                            <p>Previous versions</p>
                                <ul>
                                    <li>V1.2: Included Condensation layer that works on top of Gemini pro architecture.</li>
                                    <li>V1.1: Included Interval Frame Sampling (IFS), that extracts frames from video in a fixed interval. Leads to extraction of unfocused, noisy frames</li>
                                    <li>V1.0: Can generate captions from images using simple encoder-decoder architecture (custom).</li>
                                </ul>"""

def video_identity(video):
    local_captions, extracted_frames_count, extracted_frames = split_frames(video)

    interval_frame_sampling = {
        "Extracted frames" : extracted_frames_count,
    }

    summary = condensation(local_captions)

    generated_captions = {}
    for i in range(extracted_frames_count):
        generated_captions[str(i)] =  local_captions[i]
    
    audio = text_to_audio(summary)

    return (interval_frame_sampling, extracted_frames, generated_captions, summary, audio)

demo = gr.Interface(video_identity, 
                    inputs=gr.Video(width=400, height=400, container=True), 
                    outputs=[
                        gr.JSON(label="Extracted frames using IFS"), 
                        gr.Gallery(label="Extracted_frames"),
                        gr.JSON(label="Generated captions for extracted frames"), 
                        gr.Textbox(label = "Condensed final summary"),
                        gr.Audio(label="Summarized content as audio")
                    ],
                    examples=[["footage3.mp4"], ["ronaldo.mp4"], ["footage4.mp4"]],
                    title="Video Understanding V1.3",
                    description=interface_description,
)
demo.launch(share=True)