from azapi import AZlyrics

API_DELAY = 5


def get_lyrics(artist_known, song):
    az_api = AZlyrics()  # no google because artist known
    az_api.artist = artist_known
    az_api.title = song

    lyrics = az_api.getLyrics(sleep=API_DELAY)
    return az_api.artist, az_api.title, lyrics


def get_songs(artist_guess):
    az_api = AZlyrics('google')  # fix the guess with google
    az_api.artist = artist_guess

    return az_api.getSongs(sleep=API_DELAY)