import gradio as gr

from modules import settings, shared_options_funcs

general = settings.Section('General')
llamacpp = settings.Section('Llama.cpp')
tabbyapi = settings.Section('TabbyAPI')

templates = [
    settings.Template(general, "model_dir", '', "Model directory"),
    settings.Template(general, "model", None, "Selected model", gr.Dropdown, lambda: {"choices": shared_options_funcs.list_models(), "allow_custom_value": False}, refresh=shared_options_funcs.list_models),
    settings.Template(general, "run_at_startup", True, "Run the backend at startup", gr.Checkbox),
    settings.Template(general, "backend_startup_timeout", 30, "Startup inactivity detection timeout", gr.Number),
    settings.Template(general, "read_tensor_info", False, "Read model metadata before starting the backend", gr.Checkbox),

    settings.Template(llamacpp, "llamacpp_exe", 'llama-server', "Llamacpp executable"),
    settings.Template(llamacpp, "llamacpp_port", '8080', "Port for llamacpp to listen on"),
    settings.Template(llamacpp, "llamacpp_host", '0.0.0.0', "Host for llamacpp to listen on"),
    settings.Template(llamacpp, "llamacpp_cmdline", '', "Command line options"),
    settings.Template(llamacpp, "llamacpp_cmdline_permodel", '', "Model-specific command-line options", gr.Textbox, dict(lines=8), info="One model per line, like this: (copy model name from the main page)\nmodel.gguf: --flash-attn\nllama6.gguf: --ctx-size 4096"),

    settings.Template(tabbyapi, "tabbyapi_path", '', "Path to TabbyAPI installation dir"),
    settings.Template(tabbyapi, "tabbyapi_port", '5000', "Port for TabbyAPI to listen on"),
    settings.Template(tabbyapi, "tabbyapi_host", '0.0.0.0', "Host for TabbyAPI to listen on"),
    settings.Template(tabbyapi, "tabbyapi_cmdline", '', "Command line options"),
    settings.Template(tabbyapi, "tabbyapi_cmdline_permodel", '', "Model-specific command-line options", gr.Textbox, dict(lines=8), info="One model per line, like this: (copy model name from the main page)\nmodel-exl2: --log-prompt\nllama7-exl4: --cache-size 8192"),
]
