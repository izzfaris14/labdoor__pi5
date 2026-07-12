import tkinter as tk
import cv2
from PIL import Image, ImageTk

# ==========================================
# THE AI SWITCHBOARD
# Change this variable to "linus", "farid", or "mock" to test different scripts
ACTIVE_AI = "mock"
# ==========================================

if ACTIVE_AI == "linus":
    import linus_ai as ai_engine
elif ACTIVE_AI == "farid":
    import farid_ai as ai_engine
else:
    import mock_setup as ai_engine

class GatekeeperGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Attire Check System")

        # Lock to Kiosk Mode
        self.attributes('-fullscreen', True)
        self.bind("<Escape>", lambda e: self.destroy())  # PRESS ESC TO QUIT

        # Configure 80/20 Layout
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=4)
        self.rowconfigure(1, weight=1)

        # Video Label
        self.video_label = tk.Label(self, bg="black")
        self.video_label.grid(row=0, column=0, sticky="nsew")

        # Status Label
        self.status_label = tk.Label(self, text="System Initializing...",
                                     font=("Helvetica", 48, "bold"), bg="gray", fg="white")
        self.status_label.grid(row=1, column=0, sticky="nsew")

        # Start the video engine
        self.update_frame()

    def update_frame(self):
        # 1. Ask the engine for the newest annotated picture and text
        # It will seamlessly pull from Linus, Farid, or the mock script based on your switchboard!
        frame, status_text, color = ai_engine.get_processed_frame()

        if frame is not None:
            # Convert and display...
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            self.status_label.config(text=status_text, bg=color)

        self.after(10, self.update_frame)


if __name__ == "__main__":
    app = GatekeeperGUI()
    app.mainloop()