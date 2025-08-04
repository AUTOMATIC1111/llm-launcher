# LLM Launcher

This is a gradio UI that allows to launch server process for llama.cpp or TabbyAPI,
track its stats, restart if it crashes, select which model to use in UI, view model
properties (including layer list and templates), and download models from huggingface.

> **Note:** The code is for my personal use and there is no support. llama.cpp and/or TabbyAPI
are assumed to be already installed.

## Installation

* have python and git installed
* clone the repository and chdir to its path
* to install dependencies, run: `pip install -r requirements.txt`
* to start the program, run: `python main.py`
* after running for the first time, go to settings tab and set paths for llama.cpp/TabbyAPI.
