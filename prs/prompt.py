import re

def split_weighted_subprompts(input_string, normalize=True):
    parsed_prompts = [
        (match.group("prompt").replace("\\:", ":"), float(match.group("weight") or 1))
        for match in re.finditer(prompt_parser, input_string)
    ]
    if not normalize:
        return parsed_prompts
    # weight_sum = sum(map(lambda x: x[1], parsed_prompts))
    positive_weight_sum = sum(
        map(lambda px: px[1] if (px[1] > 0) else 0, parsed_prompts)
    )
    negative_weight_sum = -sum(
        map(lambda nx: nx[1] if (nx[1] < 0) else 0, parsed_prompts)
    )
    num_negative_weights = sum(map(lambda nx: 1 if (nx[1] < 0) else 0, parsed_prompts))
    if positive_weight_sum == 0:
        print(
            "Warning: Positive subprompt weights add up to zero. Discarding and using even weights instead."
        )
        positive_weight_sum = 1
    if negative_weight_sum == 0:
        if num_negative_weights > 0:
            print(
                "Warning: Negative subprompt weights add up to zero. Discarding and using even weights instead."
            )
        negative_weight_sum = 1
    if negative_weight_sum < 0:
        print(
            "Warning: Negative subprompt weights add up to less than one. Not normalizing."
        )
        negative_weight_sum = 1
    return [
        (
            x[0],
            (x[1] / positive_weight_sum)
            if (x[1] > 0)
            else (x[1] / negative_weight_sum),
        )
        for x in parsed_prompts
    ]


prompt_parser = re.compile(
    """
    (?P<prompt>     # capture group for 'prompt'
    (?:\\\:|[^:])+  # match one or more non ':' characters or escaped colons '\:'
    )               # end 'prompt'
    (?:             # non-capture group
    :+              # match one or more ':' characters
    (?P<weight>     # capture group for 'weight'
    -?\d+(?:\.\d+)? # match positive or negative integer or decimal number
    )?              # end weight capture group, make optional
    \s*             # strip spaces after weight
    |               # OR
    $               # else, if no ':' then match end of line
    )               # end non-capture group
""",
    re.VERBOSE,
)
