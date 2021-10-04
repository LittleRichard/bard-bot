import glob
import os


ALL_NESTED_TXT_BLOB = "**/*.txt"
ALL_NESTED_DIR_BLOB = "/**/"


def get_artist_name_from_full_path(sugg_path, base_dir):
    sugg = sugg_path.split(base_dir + '/')[1]
    return sugg[:-1]


def get_full_path_from_artist_name(artist, base_dir):
    return base_dir + '/' + artist


def get_file_save_path(artist, song_name, base_dir):
    song_name = song_name.replace('/', '_')

    # makes the artist directory if not exists
    artist_folder = get_full_path_from_artist_name(artist, base_dir)
    if not os.path.exists(artist_folder):
        os.makedirs(artist_folder)

    return (
        artist_folder
        + '/'
        + song_name + '.txt'
    )


def get_all_artist_paths(target_path):
    for path in glob.iglob(target_path + ALL_NESTED_DIR_BLOB, recursive=False):
        if path != target_path:
            yield path


def get_num_files(path):
    num_files = 0
    # i'm sure there's a more efficient way to do this. w/e for barbot
    for _ in glob.iglob(path + ALL_NESTED_TXT_BLOB, recursive=True):
        num_files += 1
    return num_files


def save_text_data(save_path, text):
    with open(save_path, 'w') as f:
        f.write(text)


def load_text_data(directory):
    text = ""
    num_files = 0
    for file_path in glob.iglob(
            directory + ALL_NESTED_TXT_BLOB,
            recursive=True):
        num_files += 1
        with open(file_path, 'r', encoding="utf-8-sig") as file:
            file_content = file.read()
            text += file_content

    return num_files, text

