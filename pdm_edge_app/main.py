from pdm_app.config_loader import ConfigLoader
from pdm_app.ui import PdmEdgeApplication


def main() -> None:
    config = ConfigLoader.load("config.json")
    app = PdmEdgeApplication(config)
    app.mainloop()


if __name__ == "__main__":
    main()
