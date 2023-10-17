from langserve.lzstring import LZString


def test_lzstring() -> None:
    s = "Žluťoučký kůň úpěl ďábelské ódy!"

    # generated with original js lib
    jsLzStringBase64 = (
        "r6ABsK6KaAD2aLCADWBfgBPQ9oCAlAZAvgDobEARlB4QAEOAjAUxAGd4BL5AZ4BMBPAQiA=="
    )
    jsLzStringBase64Json = "N4Ig5gNg9gzjCGAnAniAXKALgS0xApuiPgB7wC2ADgQASSwIogA0IA4tHACLYBu6WXASIBlFu04wAMthiYBEhgFEAdpiYYQASS6i2AWSniRURJgCCMPYfEcGAFXyJyozPBUATJB5pt8Kp3gIbAAvfB99JABrAFdKGil3MBj4MEJWcwBjRCgVZBc0EBEDIwyAIzLEfH5CrREAeRoADiaAdgBONABGdqaANltJLnwAMwVKJHgicxpyfDcAWnJouJoIJJS05hoYmHCaTCgabPx4THxZlfj1lWTU/BgaGBjMgAsaeEeuKEyAISgoFEAHSDBgifD4cwQGBQdAAbXYNlYAA0bABdAC+rDscHBhEKy0QsUoIAxZLJQA"  # noqa: E501

    compressed = LZString.compressToBase64(s)
    assert LZString.compressToBase64(s) == jsLzStringBase64
    assert LZString.decompressFromBase64(compressed) == LZString.decompressFromBase64(
        jsLzStringBase64
    )
    assert s == LZString.decompressFromBase64(compressed)

    jsonString = '{"glossary":{"title":"example glossary","GlossDiv":{"title":"S","GlossList":{"GlossEntry":{"ID":"SGML","SortAs":"SGML","GlossTerm":"Standard Generalized Markup Language","Acronym":"SGML","Abbrev":"ISO 8879:1986","GlossDef":{"para":"A meta-markup language, used to create markup languages such as DocBook.","GlossSeeAlso":["GML","XML"]},"GlossSee":"markup"}}}}}'  # noqa: E501

    compressed = LZString.compressToBase64(jsonString)
    assert compressed == jsLzStringBase64Json
    assert LZString.decompressFromBase64(compressed) == jsonString
