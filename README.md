# Monsieur Tracant

The well known robotic street artist.

Requires Python 3.9+.

## Installation

First install [vpype](https://vpype.readthedocs.io/en/latest/install.html)
```
sudo apt-get install pipx
```

There is a custom paper template we use for the HP 7576A. Therefore copy `.vpype.toml` to

* `/home/username/.vpype.toml` on Linux
* `/Users/username/.vpype.toml` on Mac
* `C:\Users\username\.vpype.toml` on Windows

See [vpype dcoumentation](https://vpype.readthedocs.io/en/latest/cookbook.html?highlight=page%20size#faq-custom-config-file) for further information

We are using `venv` to have a clean environment.

* create a new environment with `python3 -m venv mp_env` or `virtualenv mp_env --python=python3.9`
* activate environment `source mp_env/bin/activate`
* install dependencies with `pip install -r requirements.txt`
* start script with `python monsieurTracant.py`