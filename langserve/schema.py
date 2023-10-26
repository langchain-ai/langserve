try:
    from pydantic.v1 import BaseModel
except ImportError:
    from pydantic import BaseModel


class CustomUserType(BaseModel):
    """Inherit from this class to create a custom user type.

    Use a custom user type if you want the data to de-serialize
    into a pydantic model rather than the equivalent dict representation.

    In general, make sure to add a `type` attribute to your class
    to help pydantic to discriminate unions.

    https://docs.pydantic.dev/1.10/usage/types/#discriminated-unions-aka-tagged-unions

    Limitations:
        At the moment, this type only works SERVER side and is used
        to specify desired DECODING behavior. If inheriting from this type
        the server will keep the decoded type as a pydantic model instead
        of converting it into a dict.
    """
