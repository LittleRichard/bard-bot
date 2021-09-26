import glob


def load_text_data(directory):
    file_paths = glob.glob(directory + "*.txt")
    text = ""
    for file_path in file_paths:
        with open(file_path, 'r', encoding="utf-8-sig") as file:
            file_content = file.read()
            text += file_content

    return text
