# pTTS Menubar

Small menubar app to start/stop the ptts player server without Xcode.

## Build

1. Install Xcode Command Line Tools if you do not have them:
   `xcode-select --install`
2. Build the app:
   `./build.sh`

The app will be at `macos/ptts-menubar/build/pTTS Menubar.app`.

## Run

- Click the menubar item to start/stop the server. When the server responds, your default browser opens to the player UI.

Logs are written to `~/Library/Logs/ptts-menubar.log`.

On Intel Macs the app runs `./bin/pmx uv run ptts play`. On Apple Silicon it runs `uv run ptts play`.

The app auto-locates the repo by walking up from the app bundle; keep the app inside the repo. If you move the app elsewhere, set `PTTS_ROOT` to the repo root before launching.

On first launch, the app offers to create a symlink in `/Applications` so you can open it from Spotlight.
