from azapi import AZlyrics
import random

API_DELAY_MIN = 8
API_DELAY_JITTER = 4

_PROXIES = {}


def _get_sleep():
    return API_DELAY_MIN + API_DELAY_JITTER * random.random()


def get_lyrics(artist_known, song):
    az_api = AZlyrics(proxies=_PROXIES)  # no google because artist known
    az_api.artist = artist_known
    az_api.title = song

    lyrics = az_api.getLyrics(
        sleep=_get_sleep()
    )
    return az_api.artist, az_api.title, lyrics


def get_songs(artist_guess):
    az_api = AZlyrics('google', proxies=_PROXIES)  # fix the guess with google
    az_api.artist = artist_guess

    return az_api.getSongs(
        sleep=_get_sleep()
    )
