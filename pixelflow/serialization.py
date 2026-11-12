"""Reading a result type back from the plain dictionary its ``to_dict()`` wrote.

Every result type pixelflow serializes - KeyPoint, Detection, Classification - is
restored by the same two rules: take the keys its constructor accepts and leave the
rest, and share no list or dictionary with the data it was read from. Both rules live
here so the three ``from_dict`` methods cannot come to disagree about them.
"""

import functools
import inspect
from typing import Any, Dict, Tuple

__all__ = ["constructor_fields", "own"]

_CONTAINERS = (list, dict)


@functools.lru_cache(maxsize=None)
def _constructor_arguments(cls: type) -> Tuple[str, ...]:
    """The argument names `cls` is built from, read once per class."""
    return tuple(inspect.signature(cls).parameters)


def constructor_fields(cls: type, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pick out of `data` the entries that `cls`'s constructor accepts.

    The constructor is the single statement of what a result type is made of, so the
    field list is read from it rather than kept as a second copy: a field added to the
    class is restored without anyone remembering to list it. Keys the constructor does
    not know are left behind, which is what lets a dictionary written by a newer
    pixelflow still read, and keys that are absent are simply not passed, so they take
    the constructor's defaults.

    Args:
        cls (type): The class about to be constructed.
        data (Dict[str, Any]): A dictionary as written by that class's `to_dict()`.

    Returns:
        Dict[str, Any]: A new dictionary of keyword arguments for `cls`. Its values
            are still the ones in `data`; pass it through `own` to separate them.
    """
    return {key: data[key] for key in _constructor_arguments(cls) if key in data}


def own(value: Any) -> Any:
    """
    Copy a list or dictionary and every list and dictionary inside it.

    `to_dict()` hands containers such as `metadata` out by reference - it runs once
    per frame and usually feeds `json.dumps`, where a copy would be wasted - so the
    copy is made on the way back in, and a restored object never shares state with
    the one it was written from. Copying whatever is a container, not a named few
    fields, keeps that true for fields added later.

    Serialized data is lists, dictionaries and scalars, and for that shape this is
    several times faster than `copy.deepcopy`, which tracks every object it visits
    in case the structure refers back to itself. Anything that is not a container -
    a string, a number, an array a caller put in by hand - is shared, not copied.
    Scalars are tested for in the loop, not by a call that returns them unchanged:
    they are nearly all of what a result holds.

    Args:
        value: A list or dictionary of plain data.

    Returns:
        A new container of the same shape, sharing no list or dictionary with `value`.
    """
    if isinstance(value, list):
        return [own(item) if isinstance(item, _CONTAINERS) else item for item in value]
    return {key: own(item) if isinstance(item, _CONTAINERS) else item
            for key, item in value.items()}
