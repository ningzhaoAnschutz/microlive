"""Console entry point for the default MicroLive Aurora interface."""


def main():
    """Launch the default Aurora application without duplicating its setup."""
    from .app import main as launch_aurora

    launch_aurora()


if __name__ == "__main__":
    main()
