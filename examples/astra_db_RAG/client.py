from langserve import RemoteRunnable

if __name__ == '__main__':
    synonyms_runnable = RemoteRunnable("http://localhost:8000/synonyms/")
    synonyms = synonyms_runnable.invoke(input={"word": "escape"})
    print(f"Synonyms: {synonyms}")
    #
    entomology_runnable = RemoteRunnable("http://localhost:8000/entomology/")
    entomology = entomology_runnable.invoke("What order do wasps belong to?")
    print(f"Entomological fact: {entomology}")
    #
    # make it really obvious what the streamed chunks are:
    for chunk in entomology_runnable.stream("Compare Collembola and Thysanura"):
        print(f"[{chunk}]", end="", flush=True)
