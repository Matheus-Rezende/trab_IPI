import tkinter as tk

from model import ImageModel
from controller import ImageController
from view import ImageView

if __name__ == "__main__":
    root = tk.Tk()
    model = ImageModel()
    controller = ImageController(model)
    view = ImageView(root, controller)
    root.mainloop()
