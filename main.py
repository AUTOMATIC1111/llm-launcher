from modules import settings, shared, shared_options, ui_main, cmd_args, userscripts


def main():
    userscripts.load_userscripts('userscripts')

    shared.args = cmd_args.parser.parse_args()

    shared.opts = settings.Settings(shared_options.templates)
    settings_ui = settings.SettingsUi(shared.opts, shared.args.config)

    launcher = ui_main.LlmLauncher()
    ui = launcher.create_ui(settings_ui)

    ui.queue(default_concurrency_limit=10).launch(prevent_thread_lock=True, favicon_path="assets/favicon.png", allowed_paths=["assets"])

    launcher.launch_at_startup()

    ui.block_thread()


if __name__ == "__main__":
    main()

