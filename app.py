import tkinter as tk
from detection import detect_and_count
import cv2
from ffmpegcv import VideoCaptureStream as VCS
from PIL import ImageTk, Image

window = tk.Tk()
window.title("Test app")

# when detection_flag = False - app shows camera stream in real time
# when the flag = True - app shows image with detection boxes
detection_flag = False 
data = None

# label for displaying number of detections
label = tk.Label(
    window, text="", font=("Calibri 15 bold")
)
label.pack(pady=20)

# set window size
window.geometry("1000x1000")

# Camera has two channels Channel_1 - full high resolution, it is used for detection
# Channel_2 - preview low resolution, used in normal "stream" mode
cap = VCS("rtsp://user08:Mrd12345678@2.1.3.33:554/1/2")

def on_click_btn1():
    global data
    global detection_flag
    cap2 = VCS("rtsp://user08:Mrd12345678@2.1.3.33:554/1/1")
    detection_flag = not detection_flag
    if detection_flag:
        _, frame = cap2.read()
        data, num_handles = detect_and_count(frame)
        label["text"] = str(num_handles)

btn1 = tk.Button(window, text="Подсчет", command=on_click_btn1)
btn1.pack(pady=20)

stream_window = tk.Label(window)
stream_window.pack()

# define canvas dimensions
canvheight = 540
canvwidth = 960

# define video stream function
def video_stream():
    global cap
    global detection_flag
    if detection_flag:
        img = data
        img = Image.fromarray(img)
    else:
        _, frame = cap.read()
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
    img = img.resize((canvwidth, canvheight))
    imgtk = ImageTk.PhotoImage(image=img)
    stream_window.imgtk = imgtk
    stream_window.configure(image=imgtk)
    stream_window.after(1, video_stream)

# initiate video stream
video_stream()

# run the tkinter main loop
window.mainloop()