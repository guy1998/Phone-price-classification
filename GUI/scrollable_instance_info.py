import tkinter as tk
from tkinter import ttk
from GUI.input_tracker import get_continuous_features, get_categorical_features, set_user_input


class ScrollableFrame(tk.Frame):
    def __init__(self, master, **kwargs):
        tk.Frame.__init__(self, master, **kwargs)

        self.canvas = tk.Canvas(self, width=150, height=395)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side="left", fill="y", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.scrollable_frame.bind("<Configure>", self.on_frame_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        self.scrollable_frame_id = 0

    def on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_canvas_configure(self, event):
        width = event.width
        self.canvas.itemconfig(self.scrollable_frame_id, width=width)


def create_widgets_for_continuous_features(scrollable_frame):
    continuous_features = get_continuous_features()
    for feature in continuous_features:
        temp_label = ttk.Label(scrollable_frame, text=feature + ":")
        temp_label.pack()
        entry = tk.Entry(scrollable_frame)
        entry.bind("<Key>", lambda event: set_user_input(feature, event.widget.get()))
        entry.pack()


def create_widgets_for_categorical_features(scrollable_frame):
    categorical_features = get_categorical_features()
    for feature in categorical_features:
        temp_label = ttk.Label(scrollable_frame, text=feature + ":")
        temp_label.pack()
        selected_option = tk.StringVar()
        dropdown = ttk.OptionMenu(scrollable_frame, selected_option, "No", "Yes", "No",
                                  command=lambda value: set_user_input(feature, 1) if value == "Yes" else set_user_input(feature, 0))
        dropdown.config(width=15)
        dropdown.pack(pady=(0, 5))


def scrollable_part(main_frame):
    scrollable_frame = ScrollableFrame(main_frame)
    scrollable_frame.grid(row=1, column=0)
    create_widgets_for_continuous_features(scrollable_frame.scrollable_frame)
    create_widgets_for_categorical_features(scrollable_frame.scrollable_frame)
    return scrollable_frame
