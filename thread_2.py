import base64
import os
from multiprocessing import Process, Queue
import time

import cv2
import numpy as np
import pygame
import requests
from dotenv import load_dotenv
from elevenlabs import generate, play, set_api_key

import face_recognition

# from frames import add_faces

load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
set_api_key(os.environ.get("ELEVENLABS_API_KEY"))

CAPTURE_TIME_BUFFER = 5
CAPTURE_EVERY_X_FRAMES = 30 * CAPTURE_TIME_BUFFER

curr_subtitle = "--"

def play_music(track_path):
    # Initialize pygame mixer
    pygame.mixer.init()
    # Load the music file
    pygame.mixer.music.load(track_path)
    pygame.mixer.music.set_volume(0.3)
    # Play the music file indefinitely (the argument -1 means looping forever)
    pygame.mixer.music.play(-1)

    # Keep the program running to play music
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)  # You can adjust the tick rate as needed


# Process target function
def music_process():
    play_music("exit.mp3")  # Replace with your actual file path


def pass_to_gpt4_vision(base64_image, script):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "system",
                "content": """
You are the narrator of an  hero film. Narrate the characters as if you were narrating the main characters in an epic opening sequence.
Make it really awesome, while really making the characters feel epic. Don't repeat yourself. Make it short, max one line 10-20 words. Build on top of the story as you tell it. Don't use the word image. 
As you narrate, pretend there is an epic Hans Zimmer song playing in the background.
Use words that are simply but poetic, a 4th grader should be able to understand it perfectly.
Build a back story for each of the characters as the heros of a world they're trying to save.

if you clearly see their name beneath their face, refer to the character by name.

Do not invent names for characters.
          """.strip(),
            },
        ]
        + script
        + generate_new_line(base64_image),
        "max_tokens": 300,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
    print(response.json())
    gpt_4_output = response.json()["choices"][0]["message"]["content"]
    return gpt_4_output


def enhance_image_contrast_saturation(image):
    # Convert to float to prevent clipping values
    image = np.float32(image) / 255.0

    # Adjust contrast (1.0-3.0)
    contrast = 1.5
    image = cv2.pow(image, contrast)

    # Convert to HSV color space to adjust saturation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Adjust saturation (1.0-3.0)
    saturation_scale = 1.15
    hsv[:, :, 1] *= saturation_scale

    # Convert back to BGR color space
    enhanced_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Clip the values and convert back to uint8
    enhanced_image = np.clip(enhanced_image, 0, 1)
    enhanced_image = (255 * enhanced_image).astype(np.uint8)

    return enhanced_image


def play_audio(text):
    audio = generate(text, voice=os.environ.get("ELEVENLABS_VOICE_ID"))
    unique_id = base64.urlsafe_b64encode(os.urandom(30)).decode("utf-8").rstrip("=")
    dir_path = os.path.join("narration", unique_id)
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, "audio.wav")

    with open(file_path, "wb") as f:
        f.write(audio)

    play(audio)


def generate_new_line(base64_image):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe this scene like you're a narrator in a movie",
                },
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
    resized_image = cv2.resize(
        image, (max_width, new_height), interpolation=cv2.INTER_AREA
    )
    return resized_image


def add_subtitle(image, frame_count, text="", max_line_length=30):
    countdown_seconds = CAPTURE_TIME_BUFFER - (frame_count % CAPTURE_EVERY_X_FRAMES) // 30  # Convert frame count to seconds

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # White color for the main text
    shadow_color = (0, 0, 0)  # Black color for the shadow
    line_type = 2
    margin = 10  # Margin for text from the bottom
    line_spacing = 30  # Space between lines
    shadow_offset = 2  # Offset for shadow

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

        # Draw shadow
        cv2.putText(
            image,
            line,
            (text_x + shadow_offset, text_y + shadow_offset),
            font,
            font_scale,
            shadow_color,
            line_type,
        )

        # Draw main text
        cv2.putText(
            image, line, (text_x, text_y), font, font_scale, font_color, line_type
        )

    font_scale = 2
    countdown_text = f"{countdown_seconds}"
    text_size = cv2.getTextSize(countdown_text, font, font_scale, line_type)[0]
    text_x = image.shape[1] - text_size[0] - margin  # Position to the right
    text_y = margin + text_size[1]  # Position at the top

    # Draw shadow for countdown
    cv2.putText(image, countdown_text, (text_x + shadow_offset, text_y + shadow_offset), font, font_scale, shadow_color, line_type)

    # Draw countdown text
    cv2.putText(image, countdown_text, (text_x, text_y), font, font_scale, font_color, line_type)
    
    return image


def narrator_process(queue):
    while True:
        if not queue.empty():
            gpt_4_output = queue.get()
            play_audio(gpt_4_output)
            queue.put("--")
            time.sleep(2)


def process_frames(queue):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Webcam not accessible in process_frames.")
        return

    frame_count = 0
    script = []

    cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Webcam", 600, 338)

    while True:
        ret, frame = cap.read()
        original_frame = frame.copy()
        frame = cv2.flip(frame, 1)

        if not ret:
            break
        
        frame = enhance_image_contrast_saturation(frame)

        if not queue.empty():
            in_queue = queue.get()
            if in_queue == "--":
                global curr_subtitle
                curr_subtitle = "--"

        frame_with_subtitle = add_subtitle(frame, frame_count, curr_subtitle)

        frame_count += 1
        print(frame_count)
        cv2.imshow("Webcam", frame_with_subtitle)

        if frame_count % CAPTURE_EVERY_X_FRAMES == 0:
            face_locations = face_recognition.face_locations(original_frame)
            if not face_locations:
                print("no faces")
                continue
            
            print("----gpt-time----")
            resized_frame = resize_image(original_frame)
            # resized_frame = add_faces(resized_frame)
            retval, buffer = cv2.imencode(".jpg", resized_frame)
            base64_image = base64.b64encode(buffer).decode("utf-8")
            gpt_4_output = pass_to_gpt4_vision(base64_image, script)
            script = script + [{"role": "assistant", "content": gpt_4_output}]
            print(gpt_4_output)

            filename = "frame.jpg"
            cv2.imwrite(filename, resized_frame)

            curr_subtitle = gpt_4_output

            frame_count += 1
            queue.put(gpt_4_output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    queue = Queue()
    frames_process = Process(target=process_frames, args=(queue,))
    music_proc = Process(target=music_process)
    narrator_proc = Process(target=narrator_process, args=(queue,))

    frames_process.start()
    music_proc.start()
    narrator_proc.start()

    frames_process.join()
    music_proc.join()
    narrator_proc.join()


if __name__ == "__main__":
    main()
