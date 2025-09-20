#!/usr/bin/env python3.11
"""
GUI Launcher for AR Hand Control Application
This helps avoid camera permission issues on macOS
"""

import tkinter as tk
from tkinter import messagebox
import subprocess
import sys
import os

def launch_ar_app():
    """Launch the AR hand control application"""
    try:
        # Change to the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        # Launch the main application
        subprocess.run([sys.executable, "main.py"])
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to launch AR app:\n{e}")

def main():
    # Create a simple GUI launcher
    root = tk.Tk()
    root.title("AR Hand Control Launcher")
    root.geometry("400x200")
    root.resizable(False, False)
    
    # Center the window
    root.eval('tk::PlaceWindow . center')
    
    # Title
    title_label = tk.Label(root, text="AR Hand Control", 
                          font=("Arial", 18, "bold"))
    title_label.pack(pady=20)
    
    # Description
    desc_label = tk.Label(root, text="Launch the AR hand gesture application\nMake sure your camera is connected and accessible",
                         font=("Arial", 10), justify=tk.CENTER)
    desc_label.pack(pady=10)
    
    # Launch button
    launch_btn = tk.Button(root, text="Launch AR App", 
                          command=launch_ar_app,
                          font=("Arial", 12, "bold"),
                          bg="#4CAF50", fg="white",
                          padx=20, pady=10)
    launch_btn.pack(pady=20)
    
    # Instructions
    instructions = tk.Label(root, text="Note: If camera access is denied, check System Preferences → Privacy & Security → Camera",
                           font=("Arial", 8), fg="gray", wraplength=350)
    instructions.pack(pady=(0, 10))
    
    root.mainloop()

if __name__ == "__main__":
    main()
