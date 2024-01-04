import tkinter as tk
from tkinter import font
from GUI.Controller import get_model
from GUI.front_end_models import get_model_function_by_name


class RightSide(tk.Frame):
    def __init__(self, master, **kwargs):
        tk.Frame.__init__(self, master, **kwargs)

        self.frame = tk.Frame(self, width=370, height=390)
        self.title_label = tk.Label(self, text=get_model(),
                                    font=font.Font(family="Verdana", size=11, weight="bold"))
        self.title_label.grid(row=0, columnspan=4)

        self.text_about_prediction = tk.Text(self, height=20, width=40, wrap=tk.WORD)
        text = "Please enter the information about your mobile phone specification on the " + "left of the screen. The model will then take the data and according to " + "training it will provide you with the predicted price range of the mobile. " + "Price ranges are: \n1-low\n2-medium\n3-high\n4-very high\nThe model will provide " + "the accuracy that was achieved during training and testing to give a ballpark " + "value for the credibility of the prediction"
        self.text_about_prediction.insert("end", text)
        self.text_about_prediction.grid(row=2, columnspan=4, pady=10)

        self.predict_button = tk.Button(self, text='Predict', width=45, height=3,
                                        command=get_model_function_by_name(get_model()))
        self.predict_button.grid(row=4, columnspan=3)
