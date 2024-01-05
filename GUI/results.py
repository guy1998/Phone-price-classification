from tkinter import messagebox

result_space = {
    0: "low",
    1: 'medium',
    2: 'high',
    3: 'very high'
}


def create_result_prompt(result, model_accuracy):
    result = list(result)
    print(result)
    text = "The mobile phone seems to belong to the " + result_space[result[0]] + " price range!" + "\nThe accuracy of the model is " + str(model_accuracy)
    messagebox.showinfo("Results", text)