import tkinter as tk
from detection import detect_and_count, dumb_detection
from cv2 import cvtColor, COLOR_BGR2RGBA
from ffmpegcv import VideoCaptureStream as VCS
from PIL import ImageTk, Image
from tkinter import filedialog as fd
from tkinter import simpledialog as sd
from tkinter import messagebox as mb
import numpy as np
from datetime import datetime
import xlwt
import tensorflow as tf

window = tk.Tk()
window.title("Detection")

# when detection_flag = False - app shows camera stream in real time
# when the flag = True - app shows image with detection boxes
detection_flag = False
data = None
Config_path = "config.txt"
config = open(Config_path).read()
config = config.splitlines()

numbers_save_path = config[2]
filename = "test1.txt"

# label for displaying number of detections
label = tk.Label(
    window, text="", font=("Calibri 15 bold")
)
label.pack()

# set window size
window.geometry("1024x720")

# Camera has two channels Channel_1 - full high resolution, it is used for detection
# Channel_2 - preview low resolution, used in normal "stream" mode
cap2 = VCS(config[3])
f_top = tk.Frame(window)
f_bot = tk.Frame(window)
f_top.pack()
f_bot.pack()

def on_click_btn1():
    global data
    global detection_flag
    global cap2
    _, frame = cap2.read()
    detection_flag = not detection_flag
    if detection_flag:
        data, num_handles = detect_and_count(frame)
        numbers = str(num_handles)
        label["text"] = numbers
        data = Image.fromarray(data)
        _path = numbers_save_path + filename
        file = open(_path, "r")
        text = file.readlines()
        text2 = ""
        list = ["0","1","2","3","4","5","6","7","8","9"]
        for line in text:
            text2 += line + "\n"
            if line[0]=="{" and len(line) > 2:
                print(line)
                line = line.split(", ")
                for record in line:
                    dotflag = False
                    index = ""
                    count = ""
                    for i in range(len(record)):
                        char = record[i]
                        if char == ":":
                            dotflag = True
                        if char in list:
                            if dotflag:
                                count += char
                            else:
                                index += char
                    index2 = int(index)
                    count2 = int(count)
                    if (num_handles.get(int(index2)) == None):
                        num_handles.setdefault(index2, count2)
                    else:
                        num_handles[index2] += count2
        text2 += (str(datetime.now()) + """\n""")
        text2 += (numbers + """\n""")
        text2 += ("""За все время: """ + str(num_handles) + """\n""")
        file = open(_path, "w")
        file.write(text2)


def on_click_btn2():
    newWindow = tk.Toplevel(window)
    newWindow.title("Настройки")
    newWindow.geometry("500x500")
    f_1 = tk.Frame(newWindow)
    f_2 = tk.Frame(newWindow)
    f_3 = tk.Frame(newWindow)
    f_4 = tk.Frame(newWindow)
    f_1.pack()
    f_2.pack()
    f_3.pack()
    f_4.pack()

    def on_click_conf1():
        config[0] = fd.askdirectory()
        label1["text"] = config[0]

    def on_click_conf2():
        config[1] = fd.askopenfilename(filetypes=[('text Files', '*.pbtxt')])
        label2["text"] = config[1]
    def on_click_conf3():
        config[2] = fd.askdirectory()
        label3["text"] = config[2]
    def on_click_conf4():
        config[3] = sd.askstring(prompt = "new ip", title="camera IP")
        label4["text"] = config[3]
    def on_click_conf5():
        answer = mb.askokcancel(title='Confirmation',
        message='Save new config?',)

        if answer:
            txt = config[0] + "\n" + config[1] + "\n" + config[2] + "\n" + config[3]
            file = open("config.txt", "w")
            file.write(txt)

    label1 = tk.Label(f_1, text="path to model:  " + config[0], font=("Calibri 10"))
    label1.pack()
    btnConf1 = tk.Button(f_1, text="browse", command=on_click_conf1)
    btnConf1.pack()
    label2 = tk.Label(f_2, text="path to label map:  " + config[1], font=("Calibri 10"))
    label2.pack()
    btnConf2 = tk.Button(f_2, text="browse", command=on_click_conf2)
    btnConf2.pack()
    label3 = tk.Label(f_3, text="path to output:  " + config[2], font=("Calibri 10"))
    label3.pack()
    btnConf3 = tk.Button(f_3, text="browse", command=on_click_conf3)
    btnConf3.pack()
    label4 = tk.Label(f_4, text="camera ip:  " + config[3], font=("Calibri 10"))
    label4.pack()
    btnConf4 = tk.Button(f_4, text="Change", command=on_click_conf4)
    btnConf4.pack()
    btnConf5 = tk.Button(f_4, text="Save changes", command=on_click_conf5)
    btnConf5.pack()



btn1 = tk.Button(f_top, text="Подсчет", command=on_click_btn1)
btn1.pack(side=tk.LEFT)

btn2 = tk.Button(f_top, text="Настройки", command=on_click_btn2)
btn2.pack(side=tk.RIGHT)

stream_window = tk.Label(f_bot)
stream_window.pack(side=tk.BOTTOM)

# define canvas dimensions
canvheight = 540
canvwidth = 960

    # define video stream function

def video_stream():
    global cap2
    if not cap2.isOpened():
        im = Image.open(config[3])
        img = np.asarray(im)
        img = detect_and_count(img)
        img = Image.fromarray(img)
    else:
        global detection_flag
        if detection_flag:
            img = data
        else:
            _, frame = cap2.read()
            cv2image = cvtColor(frame, COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
    img = img.resize((canvwidth, canvheight))
    imgtk = ImageTk.PhotoImage(image=img)
    stream_window.imgtk = imgtk
    stream_window.configure(image=imgtk)
    stream_window.after(300, video_stream)

# initiate video stream
video_stream()

# run the tkinter main loop
window.mainloop()

