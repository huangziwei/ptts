# pTTS Menubar

Small menubar app to start/stop the ptts player server without Xcode.

## Build

1. Install Xcode Command Line Tools if you do not have them:
   `xcode-select --install`
2. Build the app:
   `./build.sh`

The app will be at `macos/ptts-menubar/build/pTTS Menubar.app`.

## Run

- Set `PTTS_ROOT` if your repo is not at `~/projects/ptts`:
  `PTTS_ROOT=/path/to/ptts open "build/pTTS Menubar.app"`
- Click the menubar item to start/stop the server.

Logs are written to `~/Library/Logs/ptts-menubar.log`.

On Intel Macs the app runs `./bin/pmx uv run ptts play`. On Apple Silicon it runs `uv run ptts play`.
