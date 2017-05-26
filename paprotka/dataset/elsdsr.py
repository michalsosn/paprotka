import paprotka.io as pio


def load_zip(path):
    sounds = list(map(pio.load_wave, pio.walk_zip(path)))
    names = list(pio.walk_zip_names(path))
    data = [sound.data for sound in sounds]
    return sounds, names, data
