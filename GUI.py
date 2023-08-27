# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 13:35:47 2023

@author: MSI
"""


import tkinter
from tkinter import *
from PIL import ImageTk, Image
from tkinter import messagebox
import pandas
from chatbot_py import get_response, bot_name, predict_class,intents
from disease_prediction import data,le

options = data.columns


options = options[:-1]

options


from tkinter import ttk, filedialog
from tkinter.filedialog import askopenfile
import os

from keras.preprocessing import image
from tensorflow.keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import numpy as np
from tkinter import PhotoImage
from PIL import ImageTk,Image  
import joblib

root =Tk()

root.title("Disease Predictor")

root.geometry("500x500")

root.configure(bg="lightblue")


def disease_prediction():
    for frame in main_frame.winfo_children():
        frame.destroy()
    value = []

    label1= Label(main_frame,text="Select your Symptoms",bg="lightblue",foreground="black",
                 font=("Arial",15,"bold"))
    label1.pack()

    for i in range(0,5,1):
        selected_option = StringVar(main_frame)
        label2= Label(main_frame,text="Symptom {j}".format(j=i+1),bg="lightblue",foreground="black",
                     font=("Arial",15,"bold"))
        label2.pack()
        option_menu1=OptionMenu(main_frame,selected_option,*options)
        option_menu1.configure(bg="grey",fg="white",font=("Arial",12),width=15)
        option_menu1.pack()
        value.append(selected_option)
        #options.remove(value[i].get())

    pred_label= Label(main_frame,text="",bg="lightblue",foreground="black",
                 font=("Arial",15,"bold"))
    pred_label.pack()
    
    pred_label2= Label(main_frame,text="",bg="lightblue",foreground="black",
                 font=("Arial",15,"bold"))
    pred_label2.pack()
    model = joblib.load("model.joblib")
    model2 = joblib.load("Decisiontree.joblib")
    def predict_disease():

        lst1 = []
        for i in value:
            res = i.get()
            lst1.append(res)
        lst = []
        for i in range(0,132):
            lst.append(0)
        df= pandas.DataFrame(index = ["values"],columns=options)
        for col in df.columns:
            if col in lst1:
                df.loc["values",col]=1
            else:
                df.loc["values",col]=0
        row_list = df.loc["values", :].values.ravel().tolist()
        row_list
        prediction = model.predict([row_list])
        prediction1 = model2.predict([row_list])
        pred_label.config(text="Predicted Disease: "+le.inverse_transform(prediction))
        pred_label2.config(text="Predicted Disease: "+le.inverse_transform(prediction1))
    my_button=Button(main_frame,text="Predict",bg="yellow",activebackground="grey",
                     borderwidth=3,font=("Arial",11,"bold"),command=predict_disease)
    my_button.pack(pady=10)




def pneumonia_prediction():
    for frame in main_frame.winfo_children():
        frame.destroy()
    model=load_model(r"C:\Users\MSI\Downloads\CNN_model (1).h5")
    label= Label(main_frame,text="Please provide your x-ray image for Pneumonia Prediction",bg="lightblue",foreground="black",
                font=("Arial",10,"bold"))
    label.pack(pady=10)
    
    # Add a Label widget
    
    
    label1 = Label(main_frame, text="", bg="lightblue",foreground="black",font=('Aerial 11',15,"bold"))
    
    label1.pack(pady=20)
    #imag = Image.open(r"C:\Users\MSI\Documents\Code\project\demo.jpg")
    #imag = imag.resize((250,250))
    #img = ImageTk.PhotoImage(imag)
    panel = Label(main_frame, bg="lightblue")
    panel.pack()
    
    
    def on_click():
       
        file = filedialog.askopenfile(mode='r', filetypes=[('Image files', '*.jpeg')])
        if file:
           filepath = os.path.abspath(file.name)
           img = image.load_img(filepath,target_size=(256,256))
           x = image.img_to_array(img)
           x = np.expand_dims(x,axis=0)
           img = preprocess_input(x)
           result = model.predict(img)
           predcition = ""
           if(result[0][0]) ==1:
               prediction = "Normal"
           else:
               prediction = "Pneumonia"
           imag = Image.open(filepath)
           imag = imag.resize((250,250))
           img = ImageTk.PhotoImage(imag)
           panel.config(image=img)
           panel.photo = img
           
           label1.config(text="Prediction: " + prediction) 
           
                 
       
    button = Button(main_frame, text="Input Image", font=('Georgia 13'),bg="yellow",foreground="black",command=lambda: on_click())
    button.pack(pady=40)  
   
def chat_bot():
    BG_GRAY = "#ABB2B9"
    BG_COLOR = "#17202A"
    TEXT_COLOR = "#EAECEE"

    FONT = "Helvetica 14"
    FONT_BOLD = "Helvetica 13 bold"
    head_label = Label(main_frame, bg=BG_COLOR, fg=TEXT_COLOR,
                           text="HealthCare Virtual Assistant", font=("Arial",11,"bold"), pady=10)
    head_label.place(relwidth=1)
        
        # tiny divider
    line = Label(main_frame, width=450, bg=BG_GRAY)
    line.place(relwidth=1, rely=0.07, relheight=0.012)
        
        # text widget
    text_widget = Text(main_frame, width=20, height=2, bg=BG_COLOR, fg=TEXT_COLOR,
                                font=FONT, padx=5, pady=5)
    text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
    text_widget.configure(cursor="arrow", state=DISABLED)
        
        # scroll bar
    scrollbar = Scrollbar(text_widget)
    scrollbar.place(relheight=1, relx=0.974)
    scrollbar.configure(command=text_widget.yview)
        
        # bottom label
    bottom_label = Label(main_frame, bg="grey", height=50)
    bottom_label.place(relwidth=1, rely=0.825)
    
        
    def _insert_message(msg, sender):
        if not msg:
            return
        
        msg_entry.delete(0, END)
        msg1 = f"{sender}: {msg}\n\n"
        text_widget.configure(state=NORMAL)
        text_widget.insert(END, msg1)
        text_widget.configure(state=DISABLED)
        
        msg2 = f"{bot_name}: {get_response(predict_class(msg),intents)}\n\n"
        text_widget.configure(state=NORMAL)
        text_widget.insert(END, msg2)
        text_widget.configure(state=DISABLED)
        
        text_widget.see(END)  

          
    def _on_enter_pressed(event):
        msg = msg_entry.get()
        _insert_message(msg, "You")
        # message entry box
    msg_entry = Entry(bottom_label, bg="#2C3E50", fg="black", font=("Arial",11,"bold"))
    msg_entry.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
    msg_entry.focus()
    msg_entry.bind("<Return>",_on_enter_pressed)
    
        # send button
    send_button = Button(bottom_label, text="Send", font=("Arial",11,"bold"), width=20, bg="grey",command=lambda:_on_enter_pressed(None))
    send_button.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)
   

    
options_frame =Frame(root,bg = "grey")
options_frame.pack(side=LEFT)
options_frame.pack_propagate(False)
options_frame.configure(width=100,height=500)


main_frame =Frame(root,highlightbackground="black",highlightthickness=2,bg="lightblue")
main_frame.pack(side=LEFT)
main_frame.pack_propagate(False)
main_frame.configure(width=500,height=500)

button1=Button(options_frame,text="Disease Prediction",font=("Bold",8),bg = "yellow",bd=0,fg="black",width=15,command=disease_prediction)
button1.place(x=2,y=50)

button2=Button(options_frame,text="Pneumonia Prediciton",font=("Bold",7),bg = "yellow",fg="black",width=15,command=pneumonia_prediction)
button2.place(x=1,y=100)

button3=Button(options_frame,text="Chatbot",font=("Bold",8),fg="black",bg = "yellow",width=15,command=chat_bot)
button3.place(x=2,y=150)

root.mainloop()



