import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np
import random

# Create the main window
root = tk.Tk()
root.title("Video Player with Object Detection")
root.geometry("1000x600")  # Adjusted size to accommodate buttons on the side
root.config(bg="#2E3B4E")  # Set background color of the main window

# Create a Frame for the title at the top
title_frame = tk.Frame(root, bg="#2E3B4E")
title_frame.pack(side="top", fill="x")

# Add a title label at the top of the frame
title_label = tk.Label(title_frame, text="Motion Object Tracking and Recognition", fg="white", bg="#2E3B4E", font=("Helvetica", 20, "bold"))
title_label.pack(pady=10)

# Create a Frame for the buttons beside the video display
button_frame = tk.Frame(root, bg="#1F2A37", bd=10, width=200)
button_frame.pack(side="left", fill="y")

# Create a Frame for the video display with a modern look
video_frame = tk.Frame(root, bg="#1F2A37")
video_frame.pack(side="right", fill="both", expand=True)

# Create a canvas widget to display the video inside the video frame
canvas = tk.Canvas(video_frame, bg="black", width=640, height=480)  # Fixed video size
canvas.pack(padx=10, pady=10)

# Generate random colors for each class
R = random.randint(0, 255)
G = random.randint(0, 255)
B = random.randint(0, 255)

# Class names for the MobileNet SSD model
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Generate random colors for each class
color = [(R, G, B) for i in CLASSES]

# Load the Caffe model for MobileNet SSD
net = cv2.dnn.readNetFromCaffe('d:/CV/deploy.prototxt', 'd:/CV/mobilenet_iter_73000.caffemodel')

# Define the function to load and play the video with object detection
def play_video():
    # Open a file dialog to choose a video file
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])

    if not video_path:
        print("No video file selected.")
        return

    # Disable the "Choose Video" button while the video is playing
    button1.config(state="disabled")

    # Open the selected video file using OpenCV
    cap = cv2.VideoCapture(video_path)

    # Flag to control video playback
    playing = True

    def update_frame():
        nonlocal playing
        if not playing:
            cap.release()
            return
        
        ret, frame = cap.read()  # Read a frame from the video
        if not ret:
            cap.release()
            return
        
        # Resize the frame to a fixed size (e.g., 640x480)
        frame_resized = cv2.resize(frame, (640, 480))
        (h, w) = frame_resized.shape[:2]

        # Prepare the image for object detection
        blob = cv2.dnn.blobFromImage(cv2.resize(frame_resized, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)

        # Perform forward pass and get detections
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                # Get the class id and bounding box (normalized)
                class_id = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw a rectangle around the detected object (fixed size box)
                cv2.rectangle(frame_resized, (startX, startY), (endX, endY), color[class_id], 4)

                # Add shadow effect to the text (black color for the shadow)
                text = CLASSES[class_id]
                cv2.putText(frame_resized, text, (startX + 12, startY - 17), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv2.LINE_AA)  # Shadow effect (black)
                cv2.putText(frame_resized, text, (startX + 10, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)  # Main text (green)

        # Convert the frame to RGB format (OpenCV uses BGR)
        frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Convert the frame to ImageTk format
        img = Image.fromarray(frame_resized)
        img_tk = ImageTk.PhotoImage(image=img)
        
        # Display the frame on the canvas
        canvas.create_image(0, 0, anchor="nw", image=img_tk)
        
        # Keep the image reference to prevent garbage collection
        canvas.img_tk = img_tk
        
        # Update the frame every 30 ms (adjust as needed for video playback speed)
        canvas.after(30, update_frame)

    # Start updating the frames from the video
    update_frame()

    # Stop the video and release the resources
    def stop_video():
        nonlocal playing
        playing = False  # Set flag to False to stop the video
        stop_button.pack_forget()  # Hide the stop button
        button1.config(state="normal")  # Re-enable the "Choose Video" button

    # Create Stop Video button again when a new video is selected
    if 'stop_button' in globals():
        stop_button.pack_forget()  # Ensure the previous stop button is hidden
    stop_button = tk.Button(button_frame, text="Stop Video", command=stop_video, bg="#FF5733", fg="white", font=("Helvetica", 16, "bold"), relief="raised", bd=3, width=20)  # Same width as "Choose Video"
    stop_button.pack(pady=20)  # Show the stop button

# Create Button 1 to start playing video with object detection
button1 = tk.Button(button_frame, text="Choose Video", command=play_video, bg="#4CAF50", fg="white", font=("Helvetica", 16, "bold"), relief="raised", bd=3, width=20)  # Same width as "Stop Video"
button1.pack(pady=20)

# Start the main loop
root.mainloop()
