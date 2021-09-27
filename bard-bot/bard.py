import cmd2
import tensorflow as tf
import sys
import time

from cmd2 import Cmd2ArgumentParser, with_argparser

from bard_tools.markov import (markov_get_dataset_and_model,
                               generate_text_from_model)
from bard_tools.storage import load_text_data
from bard_tools.text_utils import process_text


class Bard(cmd2.Cmd):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.artists_to_paths = {}
        self.model = None
        self.dataset = None

    def do_new_song(self, args):
        self.artists_to_paths = {}
        self.model = None
        self.dataset = None

    add_artist_argparser = Cmd2ArgumentParser()
    add_artist_argparser.add_argument('artist', nargs='?', help='artist name')

    @with_argparser(add_artist_argparser)
    def do_add_artist(self, args):
        # TODO: "fix" artist name and get path in utils, then fix next line
        self.artists_to_paths[args.artist] = f'text_data/{args.artist}/'

    remove_artist_argparser = Cmd2ArgumentParser()
    remove_artist_argparser.add_argument(
        'artist', nargs='?', help='artist name')

    @with_argparser(remove_artist_argparser)
    def do_remove_artist(self, args):
        # TODO: "fix" artist name and get path in utils, then fix next line
        self.artists_to_paths.pop(args.artist, None)  # default so no KeyError

    def do_list_artists(self, args):
        print(', '.join(self.artists_to_paths.keys()))

    build_model_argparser = Cmd2ArgumentParser()
    build_model_argparser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Show training stage outputs')
    # TODO: typaeahead on self.words_to_indices
    build_model_argparser.add_argument(
        '-s', '--seed',
        help='Used with verbose. Seed text for training')
    # TODO: by default, don't build over an existing model, ask to force.

    @with_argparser(build_model_argparser)
    def do_build_model(self, args):
        start_time = time.time()
        try:
            text = ''
            for artist, dir_path in sorted(self.artists_to_paths.items(),
                                           key=lambda item: item[0]):
                assert dir_path.endswith('/'), dir_path

                if args.verbose:
                    print(f'Loading text files from {dir_path}')

                raw_text = load_text_data(dir_path)
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
        help='Used with verbose. Seed text for training')
    # TODO: typaeahead on self.words_to_indices

    @with_argparser(generate_text_argparser)
    def do_generate_text(self, args):
        if self.model is None:
            print('Must build a model first')
            return

        # TODO: make args
        seed = args.seed if args.seed else self.dataset.get_random_sequence()
        print("Generating with seed:", seed, "\n")

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
                300,  # words_amount
                self.model.word_to_indices,
                self.model.input_sequence_length,
                self.model.indices_to_word,
            )

            print(f"Bard's song:\n{generated_text}\n")
        finally:
            print(f'Elapsed: {time.time() - start_time:.3f} seconds')

    def do_version(self, args):
        print("")
        print("Tensorflow version:", tf.__version__)
        print("GPU:", tf.test.gpu_device_name())
        print("")

    def do_quit(self, args):
        print('\nSee ya!\n')
        return super(Bard, self).do_quit(args)


if __name__ == '__main__':
    print("""Enter 'help' to see commands, and 'quit' to exit""")

    app = Bard()
    app.prompt = "Bard> "
    sys.exit(app.cmdloop())
