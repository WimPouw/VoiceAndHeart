# Heart-voice video script

Edit the text below, then run `python heartvoice_video.py` to regenerate
`heartvoice_coupling.mp4`. Each `## id` block is one text card shown between
animations. Inside a block:

- plain lines become the centered card text (one screen line each)
- `duration: 4.0` sets how long the card stays (seconds)
- `sub: ...` lines become the smaller, dimmer subtitle text

Do not rename the block ids; the Python looks them up by name.

## intro
duration: 6.0
Scene one

## heart-pulse
duration: 3.6
Scene two

## listen
duration: 3.8
Scene three

## did-you-hear
duration: 6.5
Scene four

## procedure
duration: 6.0
Scene five

## lissajous
duration: 5.0
Scene six

## rotate
duration: 6.5
Scene seven

## slowmo
duration: 3.6
Scene eight

## another-example
duration: 3.8
Scene nine

## expire
duration: 8.0
Scene ten

## expire-note
duration: 7.0
Scene eleven

## reference
Scene twelve
