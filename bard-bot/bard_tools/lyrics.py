from azapi import AZlyrics


def _get_az_api():
    # use google to make exact matches
    az_api = AZlyrics('google')

    return az_api


def get_lyrics(artist, song):
    # use google to make exact matches
    az_api = _get_az_api()
    az_api.artist = artist
    az_api.title = song

    lyrics = az_api.getLyrics(sleep=3)
    return az_api.artist, az_api.title, lyrics


def get_songs(artist):
    az_api = _get_az_api()
    az_api.artist = artist

    return az_api.getSongs(sleep=3)