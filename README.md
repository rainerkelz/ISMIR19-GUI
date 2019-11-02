# ISMIR19-GUI
This is GUI code to play around with invertible neural network models trained for framewise piano transcription.

- a pretrained model is provided (if you have trained better ones, please tell me!)
- train your own models using [this repository](https://github.com/rainerkelz/ISMIR19)
- read [the paper](http://arxiv.org/abs/1909.01622)

## Installation

it is recommended you first create a python 3 virtual environment

```
$ python3 -m venv ISMIR19-GUI
$ cd ISMIR19-GUI
$ source bin/activate
$ git clone https://github.com/rainerkelz/ISMIR19-GUI
$ cd ISMIR19-GUI
```

install all requirements (needs to be in stages):

```
$ pip install -r ISMIR19-GUI/requirements_00.txt
$ pip install -r ISMIR19-GUI/requirements_01.txt
```

you should have madmom version 0.17.dev0 or higher now (you can check with `pip list` what is installed where, and if it's indeed a `develop` install that points to your virtualenv)

## Data
obtain the [MAPS](http://www.tsi.telecom-paristech.fr/aao/en/2010/07/08/maps-database-a-piano-database-for-multipitch-estimation-and-automatic-transcription-of-music/) dataset

create datadirectory, and symlink to MAPS data
```
$ mkdir data
$ cd data
$ ln -s <path-to-where-MAPS-was-extracted-to> .
$ cd ..
```

### Hacky alternative

- in principle you can use any audiofile that is decodable by `madmom` (which uses ffmpeg), if you hack a new dataset it in the `audio_midi_dataset.py` that does not need the groundtruth
- i might do this myself, but right now i'm not feeling like it (at all)
- in fact, i'd be *very* happy if you could do this, and then make a PR!


## Start
```
$ python main.py
```
