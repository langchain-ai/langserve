import pytest

from langserve.playground import _get_mimetype


@pytest.mark.parametrize(
    "file_extension, expected_mimetype",
    [
        ("js", "application/javascript"),
        ("css", "text/css"),
        ("htm", "text/html"),
        ("html", "text/html"),
        ("txt", "text/plain"),  # An example of an unknown extension using guess_type
    ],
)
def test_get_mimetype(file_extension: str, expected_mimetype: str) -> None:
    # Create a filename with the given extension
    filename = f"test_file.{file_extension}"

    # Call the _get_mimetype function with the test filename
    mimetype = _get_mimetype(filename)

    # Check if the returned mimetype matches the expected one
    assert mimetype == expected_mimetype
