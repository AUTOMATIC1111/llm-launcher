from modules import settings, shared, shared_options, llamalauncher, cmd_args


def main():
    shared.args = cmd_args.parser.parse_args()

    shared.opts = settings.Settings(shared_options.temlates)
    settings_ui = settings.SettingsUi(shared.opts, shared.args.config)

    launcher = llamalauncher.LlamaServerLauncher()
    ui = launcher.create_ui(settings_ui)

    ui.queue(default_concurrency_limit=10).launch(prevent_thread_lock=True, favicon_path="assets/favicon.png")

    for _ in launcher.start_server():
        pass

    ui.block_thread()


if __name__ == "__main__":
    main()

