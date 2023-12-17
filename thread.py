import cv2
import time
from multiprocessing import Process, Queue
import os
import base64
import numpy as np
import requests
import threading
from dotenv import load_dotenv


load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")

def pass_to_gpt4_vision(base64_image, script):
  headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
  }

  payload = {
    "model": "gpt-4-vision-preview",
    "messages" :[
        {
          "role": "system",
          "content": """
          You are the narrator of an artsy hero film. Narrate the characters as if you were narrating the main characters in an epic opening sequence.
          Make it really noir and artsy, while really making the characters feel epic. Don't repeat yourself. Make it short, max one line 10-20 words. Build on top of the story as you tell it. Don't use the word image.
          """,
        },
      ]
      + script
      + generate_new_line(base64_image),
    "max_tokens": 300
  }

  response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
  print(response.json())
  gpt_4_output = response.json()["choices"][0]["message"]["content"]
  return gpt_4_output


def generate_new_line(base64_image):
  return [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this scene"},
        {
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{base64_image}",
        },
      ],
    },
  ]

def resize_image(image, max_width=500):
    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Calculate the ratio of the width and apply it to the new width
    ratio = max_width / float(width)
    new_height = int(height * ratio)

    # Resize the image
    resized_image = cv2.resize(image, (max_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image
    
def add_subtitle(image, text="", max_line_length=40):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # White color
    line_type = 2
    margin = 10  # Margin for text from the bottom
    line_spacing = 30  # Space between lines

    # Split text into multiple lines
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line + word) <= max_line_length:
            current_line += word + " "
        else:
            lines.append(current_line)
            current_line = word + " "
    lines.append(current_line)  # Add the last line

    # Calculate the starting y position
    text_height_total = line_spacing * len(lines)
    start_y = image.shape[0] - text_height_total - margin

    for i, line in enumerate(lines):
        text_size = cv2.getTextSize(line, font, font_scale, line_type)[0]
        text_x = (image.shape[1] - text_size[0]) // 2
        text_y = start_y + i * line_spacing

        cv2.putText(image, line, (text_x, text_y), font, font_scale, font_color, line_type)

    return image


def webcam_capture(queue):
    cap = cv2.VideoCapture(0)
    subtitle_text = "Default subtitle"

    if not cap.isOpened():
        print("Error: Webcam not accessible.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Check if there is a new message in the queue
        if not queue.empty():
            subtitle_text = queue.get()

        frame_with_subtitle = add_subtitle(frame, subtitle_text)
        cv2.imshow('Webcam', frame_with_subtitle)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_frames(queue):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Webcam not accessible in process_frames.")
        return

    frame_count = 0
    script = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        print("----capturing----")

        filename = "frame.jpg"
        cv2.imwrite(filename, frame)


        resized_frame = resize_image(frame)
        retval, buffer = cv2.imencode('.jpg', resized_frame)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        gpt_4_output = pass_to_gpt4_vision(base64_image, script)
        script = script + [{"role": "assistant", "content": gpt_4_output}]
        print("script:", script)
  
        frame_count += 1
        queue.put(gpt_4_output)

        time.sleep(5)  # Wait for 1 second

    cap.release()

def main():
    queue = Queue()
    webcam_process = Process(target=webcam_capture, args=(queue,))
    frames_process = Process(target=process_frames, args=(queue,))

    webcam_process.start()
    frames_process.start()

    webcam_process.join()
    frames_process.join()

if __name__ == "__main__":
    main()
