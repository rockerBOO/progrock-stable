import random

# pick a random item from the cooresponding text file
def randomizer(category):
    random.seed()
    randomizers = []
    with open(f"settings/{category}.txt", encoding="utf-8") as f:
        for line in f:
            randomizers.append(line.strip())
    random_item = random.choice(randomizers)
    return random_item


# replace anything surrounded by underscores with a random entry from the matching text file
def randomize_prompt(prompt: str) -> str:
    while "_" in prompt:
        start = prompt.index("_")
        end = prompt.index("_", start + 1)
        swap = prompt[(start + 1) : end]
        swapped = randomizer(swap)
        prompt = prompt.replace(f"_{swap}_", swapped, 1)
    # so we can still have underscores in prompts, replace any .. with _
    prompt = prompt.replace("..", "_")
    return prompt


# Dynamic value - takes ready-made possible options within a string and returns the string with an option randomly selected
# Format is "I will return <Value1|Value2|Value3> in this string"
# Which would come back as "I will return Value2 in this string" (for example)
# Optionally if a value of ^^# is first, it means to return that many dynamic values,
# so <^^2|Value1|Value2|Value3> in the above example would become:
# "I will return Value3 Value2 in this string"
# note: for now assumes a string for return. TODO return a desired type
def dynamic_value(incoming):
    if type(incoming) == str:  # we only need to do something if it's a string...
        if incoming == "auto" or incoming == "random":
            return incoming
        elif "<" in incoming:  # ...and if < is in the string...
            text = incoming
            while "<" in text:
                start = text.find("<")
                end = text.find(">")
                swap = text[(start + 1) : end]
                value = ""
                count = 1
                values = swap.split("|")
                if "^^" in values[0]:
                    count = values[0]
                    values.pop(0)
                    count = int(count[2:])
                random.shuffle(values)
                for i in range(count):
                    value = value + values[i] + " "
                value = value[:-1]  # remove final space
                text = text.replace(f"<{swap}>", value, 1)
            return text
        else:
            return incoming
    else:
        return incoming
