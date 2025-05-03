from jinja2 import TemplateError
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment
from datetime import datetime


# adapted from exllamav2 codebase
def render(template, template_vars):
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        enable_async=True,
        extensions=[loopcontrols],
    )

    def strftime_now(format):
        return datetime.now().strftime(format)

    def raise_exception(message):
        raise TemplateError(message)

    env.globals["strftime_now"] = strftime_now
    env.globals["raise_exception"] = raise_exception

    compiled_template = env.from_string(template)
    rendered = compiled_template.render(**template_vars)

    return rendered

