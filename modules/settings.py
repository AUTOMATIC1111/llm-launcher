import dataclasses
import json
import os
import gradio as gr

from modules import ui_common


@dataclasses.dataclass
class Section:
    text: str


@dataclasses.dataclass
class Template:
    section: Section
    key: str
    default: object
    label: str
    component: object = None
    component_args: dict = None
    refresh: object = None
    do_not_save: bool = False


class Settings:
    typemap = {int: float}
    data = None
    templates = None

    def __init__(self, templates: list[Template]):
        self.data = {}
        self.templates: dict[str, Template] = {x.key: x for x in templates}

    def load(self, filename):
        if os.path.exists(filename):
            with open(filename, "r") as f:
                self.data = json.load(f)

    def save(self, filelname):
        with open(filelname, "w") as f:
            json.dump(self.data, f)

    def __setattr__(self, key, value):
        if key in {"typemap", "data", "templates"} or (key not in self.data and key not in self.templates):
            return super(Settings, self).__setattr__(key, value)

        # Get the info related to the setting being changed
        info = self.templates.get(key, None)
        if info.do_not_save:
            return

        # Restrict component arguments
        comp_args = info.component_args if info else None
        if isinstance(comp_args, dict) and comp_args.get('visible', True) is False:
            raise RuntimeError(f"not possible to set '{key}' because it is restricted")

        self.data[key] = value
        return

    def __getattr__(self, item):
        if item in {"typemap", "data", "templates"}:
            return super(Settings, self).__getattribute__(item)

        if item in self.data:
            return self.data[item]

        if item in self.templates:
            return self.templates[item].default

        return super(Settings, self).__getattribute__(item)

    def set(self, key, value):
        oldval = self.data.get(key, None)
        if oldval == value:
            return False

        option = self.templates[key]
        if option.do_not_save:
            return False

        try:
            setattr(self, key, value)
        except RuntimeError:
            return False

        return True

    def same_type(self, x, y):
        if x is None or y is None:
            return True

        type_x = self.typemap.get(type(x), type(x))
        type_y = self.typemap.get(type(y), type(y))

        return type_x == type_y

    def dumpjson(self):
        d = {k: self.data.get(k, v.default) for k, v in self.templates.items()}
        return json.dumps(d)


class SettingsUi:
    def __init__(self, settings: Settings, config_filename):
        self.opts = settings
        self.config_filename = config_filename
        self.components = []
        self.component_dict = {}
        self.dummy_component = None

        self.interface = None
        self.result = None
        self.submit = None

    def create_setting_component(self, key, is_quicksettings=False):
        def fun():
            return self.opts.data[key] if key in self.opts.data else self.opts.templates[key].default

        info = self.opts.templates[key]
        t = type(info.default)

        args = info.component_args() if callable(info.component_args) else info.component_args

        if info.component is not None:
            comp = info.component
        elif t == str:
            comp = gr.Textbox
        elif t == int:
            comp = gr.Number
        elif t == bool:
            comp = gr.Checkbox
        else:
            raise Exception(f'bad options item type: {t} for key {key}')

        elem_id = f"setting_{key}"

        if info.refresh is not None:
            if is_quicksettings:
                res = comp(label=info.label, value=fun(), elem_id=elem_id, **(args or {}))
                ui_common.create_refresh_button(res, info.refresh, info.component_args, f"refresh_{key}")
            else:
                with gr.Row():
                    res = comp(label=info.label, value=fun(), elem_id=elem_id, **(args or {}))
                    ui_common.create_refresh_button(res, info.refresh, info.component_args, f"refresh_{key}")
        else:
            res = comp(label=info.label, value=fun(), elem_id=elem_id, **(args or {}))

        return res

    def run_settings(self, *args):
        changed = []

        for key, value, comp in zip(self.opts.templates.keys(), args, self.components):
            assert comp == self.dummy_component or self.opts.same_type(value, self.opts.templates[key].default), f"Bad value for setting {key}: {value}; expecting {type(self.opts.templates[key].default).__name__}"

        for key, value, comp in zip(self.opts.templates.keys(), args, self.components):
            if comp == self.dummy_component:
                continue

            if self.opts.set(key, value):
                changed.append(key)

        try:
            self.opts.save(self.config_filename)
        except RuntimeError:
            return f'{len(changed)} settings changed without save: {", ".join(changed)}.'

        return f'{len(changed)} settings changed{": " if changed else ""}{", ".join(changed)}.'

    def create_ui(self, demo):
        self.dummy_component = gr.Label(visible=False)

        with gr.Blocks(analytics_enabled=False) as settings_interface:
            with gr.Row():
                with gr.Column(scale=6):
                    self.submit = gr.Button(value="Apply settings", variant='primary', elem_id="settings_submit")

            self.result = gr.HTML(elem_id="settings_result")

            for i, (k, item) in enumerate(self.opts.templates.items()):
                component = self.create_setting_component(k)
                self.component_dict[k] = component
                self.components.append(component)

        self.submit.click(
            fn=self.run_settings,
            inputs=self.components,
            outputs=[self.result],
        )

        component_keys = [k for k in self.opts.templates.keys() if k in self.component_dict]

        def get_value_for_setting(key):
            value = getattr(self.opts, key)

            info = self.opts.templates[key]
            args = info.component_args() if callable(info.component_args) else info.component_args or {}

            return gr.update(value=value, **args)

        def get_settings_values():
            return [get_value_for_setting(key) for key in component_keys]

        demo.load(
            fn=get_settings_values,
            inputs=[],
            outputs=[self.component_dict[k] for k in component_keys],
            queue=False,
        )

        self.interface = settings_interface
