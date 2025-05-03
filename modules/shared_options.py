import os
import gradio as gr

from modules import settings, shared_options_funcs

llamacpp = settings.Section('Llama.cpp')

temlates = [
    settings.Template(llamacpp, "llamacpp_model_dir", '', "Model directory"),
    settings.Template(llamacpp, "llamacpp_model", None, "Selected model", gr.Dropdown, lambda: {"choices": shared_options_funcs.list_llamacpp_models(), "allow_custom_value": True}, refresh=shared_options_funcs.list_llamacpp_models),
    settings.Template(llamacpp, "llamacpp_exe", 'llama-server', "Llamacpp executable"),
    settings.Template(llamacpp, "llamacpp_port", '8080', "Port for llamacpp to listen on"),
    settings.Template(llamacpp, "llamacpp_host", '0.0.0.0', "Host for llamacpp to listen on"),
    settings.Template(llamacpp, "llamacpp_cmdline", '', "Command line options"),
    settings.Template(llamacpp, "llamacpp_cmdline_permodel", '', "Model-specific command-line options", gr.Textbox, dict(lines=8), info="One model per line, like this: (copy model name from the field above)\nmodel.gguf: --flash-attn\nllama6.gguf: --ctx-size 4096"),
    settings.Template(llamacpp, "llamacpp_startup_timeout", 30, "Startup inactivity detection timeout", gr.Number),

]
