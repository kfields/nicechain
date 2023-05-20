# nicechain :smile: :link:

AI chat app built with [NiceGUI](https://nicegui.io) and [LangChain](https://blog.langchain.dev)

-----

**Table of Contents**

- [nicechain :smile: :link:](#nicechain-smile-link)
  - [Configuration](#configuration)
  - [Installation](#installation)
    - [Recommended method using Hatch](#recommended-method-using-hatch)
      - [Best way to install Hatch if you don't have it](#best-way-to-install-hatch-if-you-dont-have-it)
  - [Usage](#usage)
  - [License](#license)

## Configuration

```console
cp .env.example .env
```
Edit the .env file and set the values for `OPENAI_API_KEY` and `COHERE_API_KEY`

## Installation

### Recommended method using [Hatch](https://github.com/pypa/hatch)
```console
pip install --upgrade pip

python3 -m pip install --user pipx
python3 -m pipx ensurepath

pipx install hatch

hatch shell
```

#### Best way to install Hatch if you don't have it
```console
pip install --upgrade pip

python3 -m pip install --user pipx
python3 -m pipx ensurepath

pipx install hatch
```

## Usage

```console
python main.py
```

## License

`nicechain` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
