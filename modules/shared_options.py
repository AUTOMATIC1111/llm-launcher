import os
import gradio as gr

from modules import settings, shared_options_funcs

llamacpp = settings.Section('Llama.cpp')

temlates = [
    settings.Template(llamacpp, "llamacpp_exe", 'llamacpp-server', "Server name"),
    settings.Template(llamacpp, "llamacpp_cmdline", '', "Command line options"),
    settings.Template(llamacpp, "llamacpp_model_dir", '', "Model directory"),
    settings.Template(llamacpp, "llamacpp_model", None, "Selected model", gr.Dropdown, lambda: {"choices": shared_options_funcs.list_llamacpp_models()}, refresh=shared_options_funcs.list_llamacpp_models),
]
