import glob


def get_artist_name_from_full_path(sugg_path, base_dir):
    sugg = sugg_path.split(base_dir + '/')[1]
    return sugg[:-1]


def load_text_data(directory):
    text = ""
    num_files = 0
    for file_path in glob.iglob(directory + "**/*.txt", recursive=True):
        num_files += 1
        with open(file_path, 'r', encoding="utf-8-sig") as file:
            file_content = file.read()
            text += file_content

    return num_files, text

