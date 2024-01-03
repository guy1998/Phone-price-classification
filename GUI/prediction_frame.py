import tkinter as tk
from tkinter import font
from GUI.Controller import get_model
from GUI.Controller import change_state, change_true
from GUI.scrollable_instance_info import scrollable_part
from GUI.right_side_of_the_frame import RightSide


def prediction_menu():

    def switching_states():
        change_state(1)
        change_true()

    frame = tk.Frame(width=650, height=440)

    back_button = tk.Button(frame, text="back", command=switching_states, width=20)
    back_button.grid(row=0, column=0)

    scrollable_part(frame)
    right_side = RightSide(frame)
    right_side.grid(row=0, column=1, rowspan=2)

    return frame
