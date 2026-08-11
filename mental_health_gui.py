"""
Student Mental Health Predictor - Desktop GUI (Tkinter)
=========================================================
Loads the SVM model trained by train_model.py and lets the user enter a
student's profile to predict the likelihood of Depression.

Run:
    python3 mental_health_gui.py

Requires model.joblib and metadata.json in the same folder
(produced by train_model.py).
"""

import json
import os
import tkinter as tk
from tkinter import messagebox, ttk

import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")
META_PATH = os.path.join(BASE_DIR, "metadata.json")

model = joblib.load(MODEL_PATH)
with open(META_PATH) as f:
    meta = json.load(f)

OPTIONS = meta["options"]
FEATURES = meta["feature_columns"]
AGE_MIN, AGE_MAX = meta["age_range"]

YES_NO = ["Yes", "No"]


class MentalHealthApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Student Mental Health Predictor - SVM")
        self.geometry("460x560")
        self.resizable(False, False)
        self.configure(bg="#f4f6f8")

        self.vars = {}
        self._build_form()
        self._build_result_area()

    def _build_form(self):
        header = tk.Label(
            self, text="Depression Risk Predictor (SVM)",
            font=("Segoe UI", 15, "bold"), bg="#f4f6f8", fg="#1f2d3d",
        )
        header.pack(pady=(16, 4))

        sub = tk.Label(
            self, text="Fill in the student profile and click Predict",
            font=("Segoe UI", 9), bg="#f4f6f8", fg="#5a6b7b",
        )
        sub.pack(pady=(0, 12))

        form = tk.Frame(self, bg="#f4f6f8")
        form.pack(padx=24, fill="x")

        row = 0
        row = self._add_dropdown(form, row, "Choose your gender", OPTIONS["Choose your gender"])
        row = self._add_spinbox(form, row, "Age", AGE_MIN, AGE_MAX)
        row = self._add_dropdown(form, row, "Your current year of Study", OPTIONS["Your current year of Study"])
        row = self._add_dropdown(form, row, "What is your CGPA?", OPTIONS["What is your CGPA?"])
        row = self._add_dropdown(form, row, "Marital status", OPTIONS["Marital status"])
        row = self._add_dropdown(form, row, "Do you have Anxiety?", OPTIONS["Do you have Anxiety?"])
        row = self._add_dropdown(form, row, "Do you have Panic attack?", OPTIONS["Do you have Panic attack?"])
        row = self._add_dropdown(
            form, row, "Did you seek any specialist for a treatment?",
            OPTIONS["Did you seek any specialist for a treatment?"],
        )

        predict_btn = tk.Button(
            self, text="Predict", command=self.predict,
            bg="#2f6fed", fg="white", font=("Segoe UI", 11, "bold"),
            relief="flat", padx=14, pady=8, cursor="hand2",
        )
        predict_btn.pack(pady=16)

    def _add_dropdown(self, parent, row, label, options):
        tk.Label(parent, text=label, bg="#f4f6f8", anchor="w",
                  font=("Segoe UI", 9)).grid(row=row, column=0, sticky="w", pady=4)
        var = tk.StringVar(value=options[0])
        cb = ttk.Combobox(parent, textvariable=var, values=options, state="readonly", width=18)
        cb.grid(row=row, column=1, sticky="e", pady=4)
        self.vars[label] = var
        return row + 1

    def _add_spinbox(self, parent, row, label, lo, hi):
        tk.Label(parent, text=label, bg="#f4f6f8", anchor="w",
                  font=("Segoe UI", 9)).grid(row=row, column=0, sticky="w", pady=4)
        var = tk.IntVar(value=int((lo + hi) / 2))
        sb = tk.Spinbox(parent, from_=lo, to=hi, textvariable=var, width=20)
        sb.grid(row=row, column=1, sticky="e", pady=4)
        self.vars[label] = var
        return row + 1

    def _build_result_area(self):
        self.result_frame = tk.Frame(self, bg="#f4f6f8")
        self.result_frame.pack(pady=10, fill="x", padx=24)

        self.result_label = tk.Label(
            self.result_frame, text="", font=("Segoe UI", 12, "bold"),
            bg="#f4f6f8",
        )
        self.result_label.pack()

        self.prob_label = tk.Label(
            self.result_frame, text="", font=("Segoe UI", 9), bg="#f4f6f8", fg="#5a6b7b",
        )
        self.prob_label.pack()

    def predict(self):
        try:
            row = {feat: self.vars[feat].get() for feat in FEATURES}
            X = pd.DataFrame([row])[FEATURES]
            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0][1]  # probability of "Depression = Yes"
        except Exception as exc:
            messagebox.showerror("Prediction error", str(exc))
            return

        if pred == 1:
            self.result_label.config(text="Result: Likely Depression", fg="#c0392b")
        else:
            self.result_label.config(text="Result: Unlikely Depression", fg="#1e8449")
        self.prob_label.config(text=f"Model confidence (Depression = Yes): {proba*100:.1f}%")


if __name__ == "__main__":
    app = MentalHealthApp()
    app.mainloop()
