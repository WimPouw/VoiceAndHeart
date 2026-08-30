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
When people vocalize at a steady tone,
each heart beat slightly changes the loudness of the voice.
sub: The heart leaves a signature on the voice.

## heart-pulse
duration: 3.6
This is the heart pulse of a participant.

## listen
duration: 3.8
Now listen to the voice
while watching the heart pulse.

## did-you-hear
duration: 6.5
Did you hear it?  Maybe not.
But plotting the heart pulse against the loudness of the voice
reveals shared hills and valleys, aligned with the heart's beat.

## procedure
duration: 6.0
We first remove the overall trend of the vocal amplitude,
and then express any deviations from that trend:
these capture the valleys and hills that relate to the heart.

## lissajous
duration: 5.0
To show the coupling, we draw a Lissajous figure:
the voice traced against the heart, moment by moment.

## rotate
duration: 6.5
The loops show the voice's loudness riding on the heart's pulse.
Perhaps you start to hear it too.
Let's rotate the figure, so the voice runs along the x-axis.

## slowmo
duration: 3.6
And once more,
in slow motion.

## another-example
duration: 3.8
Here is another example from a different trial.

## expire
duration: 8.0
The heart's signature is not only in the voice.
We find the same coupling when people simply breathe out air through the lips.
This finding suggests a mechanism of heart-voice coupling
via the lungs, rather than the voice box.

## expire-note
duration: 7.0
While this is exquisitely difficult to hear,
we obtained very reliable signatures of the heart beat
in the breathing expiration noises as well.

## reference
duration: 9.0
Hafkamp, M. S., Werner, R., Drijvers, L., Selen, L., & Pouw, W. (2025).
Dissociating Mechanisms of Heart–Voice Coupling.
Forthcoming in the Journal of the Acoustical Society of America.
sub: m.hafkamp@tilburguniversity.edu   ·   w.pouw@tilburguniversity.edu
