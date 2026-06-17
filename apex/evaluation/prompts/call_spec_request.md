Ask for executable call specifications, not guessed oracle values.

For each proposed test, return:
- the focal module and callable name
- JSON-serializable positional and keyword arguments
- the behavior class to observe (`value`, `exception`, `property`, or `non_deterministic`)

Do not invent expected return values. Apex will execute the call in the
repository workdir and synthesize assertions from the observed result.
