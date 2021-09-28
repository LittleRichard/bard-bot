import glob
import os


def get_artist_name_from_full_path(sugg_path, base_dir):
    sugg = sugg_path.split(base_dir + '/')[1]
    return sugg[:-1]


def get_full_path_from_artist_name(artist, base_dir):
    return base_dir + '/' + artist


def get_file_save_path(artist, song_name, base_dir):
    # makes the artist directory if not exists
    artist_folder = get_full_path_from_artist_name(artist, base_dir)
    if not os.path.exists(artist_folder):
        os.makedirs(artist_folder)

    return (
        artist_folder
        + '/'
        + song_name + '.txt'
    )


def save_text_data(save_path, text):
    with open(save_path, 'w') as f:
        f.write(text)


def load_text_data(directory):
    text = ""
    num_files = 0
    for file_path in glob.iglob(directory + "**/*.txt", recursive=True):
        num_files += 1
        with open(file_path, 'r', encoding="utf-8-sig") as file:
            file_content = file.read()
            text += file_content

    return num_files, text

