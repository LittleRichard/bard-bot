from azapi import AZlyrics

API_DELAY = 10

_PROXIES = {}


def get_lyrics(artist_known, song):
    az_api = AZlyrics(proxies=_PROXIES)  # no google because artist known
    az_api.artist = artist_known
    az_api.title = song

    lyrics = az_api.getLyrics(sleep=API_DELAY)
    return az_api.artist, az_api.title, lyrics


def get_songs(artist_guess):
    az_api = AZlyrics('google', proxies=_PROXIES)  # fix the guess with google
    az_api.artist = artist_guess

    return az_api.getSongs(sleep=API_DELAY)
