import string

characters_to_remove = (
    '–—”„…«»‘’“°ſ†•✠' + '!\"#$%&\'()*+-/:;<=>?@[\]^_`{|}~'
    + string.digits
)
characters_to_translate = 'ąćęłńóśźżäöüæèêéôâáà£çëîñòùúûāœï'
replacement_characters = 'acelnoszzaoueeeeoaaaeceinouuuaei'


class DataProcessor:
    def __init__(self, chars_to_remove, chars_to_translate, replacement_chars):
        self.chars_to_remove = chars_to_remove
        self.chars_to_translate = chars_to_translate
        self.replacement_chars = replacement_chars

    def preprocess_data(self, preproc_text):
        removal_translator = str.maketrans("", "", self.chars_to_remove)
        special_characters_translator = str.maketrans(
            self.chars_to_translate, self.replacement_chars, '')
        preproc_text = (
            preproc_text.lower()
            .translate(removal_translator)
            .translate(special_characters_translator)
        )
        preproc_text = "".join( list( map(
            DataProcessor.__split_punctuation_from_sentence, preproc_text)))
        preproc_text = " ".join(preproc_text.split())
        return preproc_text

    @staticmethod
    def __split_punctuation_from_sentence(char):
        if char == '.' or char == ',':
            return " " + char + " "
        else:
            return char


DATA_PROCESSOR = DataProcessor(
    characters_to_remove,
    characters_to_translate,
    replacement_characters
)


def process_text(input_text):
    return DATA_PROCESSOR.preprocess_data(input_text)


def get_text_stats(input_text):
    # lots of string stuff, probably slow at high scale
    #
    # returns (unique chars, all words in order, unique words)
    unique_characters = sorted(list(set(input_text)))
    words = input_text.split()
    vocab = sorted(set(words))

    return unique_characters, words, vocab
