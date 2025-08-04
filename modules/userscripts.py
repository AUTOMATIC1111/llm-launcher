import importlib
import importlib.util
import pathlib

from modules import errors

userscripts = []
on_app_init = []
on_app_ui = []


def load_userscripts(dirname):
    userscripts.clear()

    dirobj = pathlib.Path(dirname)
    if not dirobj.exists():
        return

    for filename in dirobj.glob('*.py'):

        module_name = filename.stem
        spec = importlib.util.spec_from_file_location(module_name, filename)

        try:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            userscripts.append(module)
        except Exception as e:
            errors.display(e, task=f'loading {filename}', full_traceback=True)
