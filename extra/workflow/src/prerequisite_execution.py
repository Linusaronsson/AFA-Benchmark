"""Guard prerequisite script arguments against bypassing resolved execution."""

import shlex


def checked_prerequisite_params(params: str, label: str) -> str:
    for argument in shlex.split(params):
        if argument.split("=", 1)[0].lstrip("+~") == "device":
            message = f"{label} cannot set device; use execution"
            raise ValueError(message)
    return params
