# Stero vision stream

This program runs both the left camera feed and right feed to a website browser.

## The protocol

Booster K1 stereo camera browser viewer (ROS 2 -> MJPEG over SSH).

## Uploading the script

Uploading the script to the Booster K1

`scp ~/Downloads/k1_stereo_viewer.py \
  booster@<Ip_Address>>:/home/booster/k1-stereo/`

## How to run the script

Nagivate to the directory
`cd /home/booster/k1-stereo`

Activate the virtual environment

`source .venv/bin/activate`

Activate the script

`python k1_stereo_viewer.py --fps 30`

## How to connect to the vision stream

`~ssh -N -L 8765:127.0.0.1:8765 booster@10.1.2.2~`

Then open on: [http://localhost:8765](http://localhost:8765)

This is a _preview_ of each camera's most recent frame, not timestamp-synchronised
stereo data for depth estimation.
