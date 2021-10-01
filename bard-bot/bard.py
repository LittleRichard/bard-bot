import os
import random

import cmd2
import tensorflow as tf
import sys
import time

from cmd2 import Cmd2ArgumentParser, with_argparser

from bard_tools.lyrics import get_lyrics, get_songs
from bard_tools.markov import (markov_get_dataset_and_model,
                               generate_text_from_model)
from bard_tools.storage import (load_text_data, get_artist_name_from_full_path,
                                get_file_save_path, save_text_data)
from bard_tools.text_utils import process_text


DEFAULT_TEXT_DATA_DIR = (
    os.path.dirname(os.path.abspath(__file__))
    + '/text_data'
)


class BardBot(cmd2.Cmd):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.artists_to_paths = {}
        self.model = None
        self.dataset = None
        self.text_data_dir = DEFAULT_TEXT_DATA_DIR

    def do_new_song(self, args):
        self.artists_to_paths = {}
        self.model = None
        self.dataset = None

    download_artist_argparser = Cmd2ArgumentParser()
    download_artist_argparser.add_argument(
        'artists',
        nargs='+',
        help='Artist name(s). Use quotes to include spaces in an artist name'
    )

    @with_argparser(download_artist_argparser)
    def do_download_artist(self, args):
        for artist in args.artists:
            song_names = list(get_songs(artist).keys())
            print(f'Found {len(song_names)} for search: {artist}')
            artist_official = None

            if len(song_names) == 0:
                print('No songs')
                return

            artist_official = None
            random.shuffle(song_names)  # unpredictable fetch
            for idx, song_name in enumerate(song_names):
                # at most, re-download 1 song to get the official
                # artist before knowing how to use cache for the rest
                if artist_official:
                    save_path = get_file_save_path(
                        artist_official, song_name, self.text_data_dir)
                    if os.path.isfile(save_path):
                        print(f'Cache has {idx+1} of {len(song_names)} : '
                              f'{artist_official} : {song_name}')
                        continue

                (
                    artist_official,
                    song_official,
                    lyrics
                ) = get_lyrics(artist, song_name)

                if not isinstance(lyrics, str):
                    print(f'Error with song {song_official}')
                    continue

                print(f'Downloaded {idx+1} of {len(song_names)} : '
                      f'{artist_official} : {song_official} : '
                      f'{len(lyrics)} chars')

                save_path = get_file_save_path(
                    artist_official, song_name, self.text_data_dir)
                if os.path.isfile(save_path):
                    print(f'Cache has {idx + 1} of {len(song_names)} '
                          f': {song_name}')

                save_text_data(save_path, lyrics)

            if artist_official:
                print(f'Official artist name: {artist_official}')
            else:
                print(f'No artist name found for {artist}')

    # like "do_" prefix, identifies for completion of the function name suffix
    def complete_download_artist(self, text, line, begidx, endidx):
        sugg_paths = self.path_complete(
            self.text_data_dir + '/' + text,
            line,
            begidx,
            endidx,
            path_filter=os.path.isdir
        )

        scrubbed_suggestions = []
        for sugg_path in sugg_paths:
            # get the directory name without full path
            sugg = get_artist_name_from_full_path(sugg_path,
                                                  self.text_data_dir)
            scrubbed_suggestions.append(sugg)

        return sorted(scrubbed_suggestions)

    add_artist_argparser = Cmd2ArgumentParser()
    add_artist_argparser.add_argument(
        'artist',
        nargs='?',
        help='Artist name to add. '
    )

    @with_argparser(add_artist_argparser)
    def do_add_artist(self, args):
        if args.artist.endswith('/'):
            print('Try again without trailing /')
        self.artists_to_paths[args.artist] = (
            f'{self.text_data_dir}/{args.artist}/'
        )

    # like "do_" prefix, identifies for completion of the function name suffix
    def complete_add_artist(self, text, line, begidx, endidx):
        return self.complete_download_artist(text, line, begidx, endidx)

    remove_artist_argparser = Cmd2ArgumentParser()
    remove_artist_argparser.add_argument(
        'artist',
        nargs='?',
        help='Artist name to remove')

    @with_argparser(remove_artist_argparser)
    def do_remove_artist(self, args):
        self.artists_to_paths.pop(args.artist, None)  # default so no KeyError

    # like "do_" prefix, identifies for completion of the function name suffix
    def complete_remove_artist(self, text, line, begidx, endidx):
        return sorted(
            self.basic_complete(
                text,
                line,
                begidx,
                endidx,
                self.artists_to_paths.keys(),
            )
        )

    def do_list_artists(self, args):
        print('Currently selected:\n')
        print('\n'.join(sorted(
            get_artist_name_from_full_path(path, self.text_data_dir)
            for path in self.artists_to_paths.values()
        )))
        print('\n')

    build_model_argparser = Cmd2ArgumentParser()
    build_model_argparser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Show training stage outputs')
    build_model_argparser.add_argument(
        '-s', '--seed',
        nargs='?',
        help='Used with verbose. Seed text for training')
    build_model_argparser.add_argument(
        '-f', '--force', action='store_true',
        help='Override and build a new model over an existing one')

    @with_argparser(build_model_argparser)
    def do_build_model(self, args):
        if self.model is not None and not args.force:
            print('Must use force option to build over a built model')
            return

        start_time = time.time()
        try:
            text = ''
            for artist, dir_path in sorted(self.artists_to_paths.items(),
                                           key=lambda item: item[0]):
                assert dir_path.endswith('/'), dir_path

                if args.verbose:
                    print(f'Loading text files from {dir_path}')

                num_files, raw_text = load_text_data(dir_path)
                if args.verbose:
                    print(f'Loaded {num_files} files from {dir_path}')

                text += process_text(raw_text)

            self.model, self.dataset = markov_get_dataset_and_model(
                text,
                show_training_stage_test=args.verbose,
                training_stage_test_seed=args.seed,
                send_output=print,
            )
        finally:
            print(f'Elapsed: {time.time() - start_time:.3f} seconds')

    generate_text_argparser = Cmd2ArgumentParser()
    generate_text_argparser.add_argument(
        '-s', '--seed',
        nargs='?',
        help='Seed text for generating text')

    @with_argparser(generate_text_argparser)
    def do_generate_text(self, args):
        if self.model is None:
            print('Must build a model first')
            return

        if args.seed and args.seed not in self.dataset.text_sequences:
            print(f'Given seed not recognized: {args.seed}')
            return

        if args.seed is None:
            seed = self.dataset.get_random_sequence()
            print(f'Generating with seed: {seed}\n')

        start_time = time.time()
        try:
            seed_for_epochs = (
                args.seed
                if args.seed
                else self.dataset.get_random_sequence()
            )

            generated_text = generate_text_from_model(
                self.model.model,
                seed_for_epochs,
                200,  # words_amount
                self.model.word_to_indices,
                self.model.input_sequence_length,
                self.model.indices_to_word,
            )

            print(f"BardBot's song:\n\n{generated_text}\n")
        finally:
            print(f'Elapsed: {time.time() - start_time:.3f} seconds')

    # like "do_" prefix, identifies for completion of the function name suffix
    def complete_generate_text(self, text, line, begidx, endidx):
        return sorted(
            self.basic_complete(
                text,
                line,
                begidx,
                endidx,
                self.dataset.text_sequences,
            )
        )

    def do_version(self, args):
        print("")
        print("Tensorflow version:", tf.__version__)
        print("GPU:", tf.test.gpu_device_name())
        print("")

    def do_quit(self, args):
        print('\nSee ya!\n')
        return super(BardBot, self).do_quit(args)


if __name__ == '__main__':
    print("""Enter 'help' to see commands, and 'quit' to exit""")

    app = BardBot()
    app.prompt = "BardBot > "
    sys.exit(app.cmdloop())
