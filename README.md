# play-theme
Use facial recognition technology to automatically play your preferred theme song when you enter the room.

## Quickstart (Mac-only)
```sh
# first ensure xcode and homebrew are installed
# then install library dependencies with homebrew
brew install python3 cmake boost-python

# start virtualenv and install Python dependencies
python3 -m venv ./venv; source venv/bin/activate;
pip3 install -r requirements.txt  # this will take a while...

# execute
python3 -m play_theme
```
