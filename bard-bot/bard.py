import cmd2
import tensorflow as tf
import sys
import time

from bard_tools.markov import (markov_get_dataset_and_model,
                               generate_text_from_model)
from bard_tools.storage import load_text_data
from bard_tools.text_utils import process_text


class Bard(cmd2.Cmd):

    def do_markov(self, args):
        directory_paths = ['text_data/']
        start_time = time.time()

        try:
            text = ''
            for dir_path in directory_paths:
                assert dir_path.endswith('/'), dir_path

                print(f'Loading {dir_path}')
                raw_text = load_text_data(dir_path)
                text += process_text(raw_text)

            model, dataset = markov_get_dataset_and_model(
                text,
                show_training_stage_test=True,
                send_output=print,
            )

            seed = dataset.get_random_sequence()
            print("Generating with seed:", seed, "\n")
            generated_text = generate_text_from_model(
                model.model,
                seed,
                words_amount=300
            )
            print(generated_text)
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
