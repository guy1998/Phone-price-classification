import tkinter as tk
from GUI.main_app import main_menu
from GUI.Controller import get_state, change_false, get_changed
from GUI.prediction_frame import prediction_menu


def update_frame(window, main_frame):
    state = get_state()
    changed = get_changed()
    if changed:
        if state == 1:
            new_frame = main_menu()
        else:
            new_frame = prediction_menu()
        change_false()
        main_frame.destroy()
        new_frame.pack()
        window.after(300, update_frame, window, new_frame)
    else:
        window.after(300, update_frame, window, main_frame)


def stage():
    window = tk.Tk()
    window.title("Main page")
    window.geometry('520x440')
    window.configure()
    main_frame = main_menu()
    main_frame.pack()
    window.resizable(False, False)
    window.after(300, update_frame, window, main_frame)
    window.mainloop()
