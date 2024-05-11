WRAPPER_DB_NAME = "wrapper_db"


def write_origin_substitution_pairs_to_db(tuples: list, file_path="substitutions.txt"):  # (origin, substitution)
    with open(file_path, "w") as file:
        for a, b in tuples:
            file.write(f"{a.lower()}, {b}\n") # set a to lower-case for convenient comparisons; do not set b to lower case as there might be capital letters
    print("writing finished!")


def read_substitutions_from_file(file_path):
    substitutions = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(', ')
            if len(parts) == 2:
                key, value = parts
                substitutions[key] = value
    return substitutions


if __name__ == '__main__':
    pairs = [("cheap", "economical"), ("addicted", "habitual use"), ("poor", "low-income")]
    write_origin_substitution_pairs_to_db(pairs)
