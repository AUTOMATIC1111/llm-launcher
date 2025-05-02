from modules import settings, shared, shared_options, llamalauncher


def main():
    shared.opts = settings.Settings(shared_options.temlates)
    settings_ui = settings.SettingsUi(shared.opts, shared.config_filename)

    launcher = llamalauncher.LlamaServerLauncher()
    ui = launcher.create_ui(settings_ui)

    ui.queue().launch(prevent_thread_lock=True, favicon_path="favicon.png")

    for _ in launcher.start_server():
        pass

    ui.block_thread()


if __name__ == "__main__":
    main()

