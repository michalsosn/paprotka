import numpy as np
import pyaudio as pa
import wave


PYAUDIO_TO_NUMPY = {
    pa.paInt8: np.int8,
    pa.paUInt8: np.uint8,
    pa.paInt16: np.int16,
    pa.paInt32: np.int32,
    pa.paFloat32: np.float32
}

NUMPY_TO_PYAUDIO = {val: key for key, val in PYAUDIO_TO_NUMPY.items()}


class Sound:
    def __init__(self, pyaudio_format, channels, rate, sample_width, data):
        self.pyaudio_format = pyaudio_format
        self.channels = channels
        self.rate = rate
        self.sample_width = sample_width
        self.data = data


def record_sound(seconds, dtype=np.int16, channels=1, rate=44100, chunk_size=1024):
    pyaudio_format = NUMPY_TO_PYAUDIO[dtype]

    audio = pa.PyAudio()
    stream = audio.open(format=pyaudio_format, channels=channels, rate=rate,
                        input=True, frames_per_buffer=chunk_size)

    parts = []
    for i in range(0, rate * seconds, chunk_size):
        part_bytes = stream.read(chunk_size, exception_on_overflow=False)
        part = np.frombuffer(part_bytes, dtype=dtype)
        parts.append(part)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    sample_width = pa.get_sample_size(pyaudio_format)
    data = np.concatenate(parts)
    return Sound(pyaudio_format, channels, rate, sample_width, data)


def play_sound(sound):
    audio = pa.PyAudio()
    stream = audio.open(format=sound.pyaudio_format, channels=sound.channels,
                        rate=sound.rate, output=True)

    stream.write(sound.data.tobytes())

    stream.stop_stream()
    stream.close()
    audio.terminate()


def save_wave(path, sound):
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(sound.channels)
        wf.setsampwidth(sound.sample_width)
        wf.setframerate(sound.rate)
        wf.writeframes(sound.data.tobytes())


def load_wave(path, chunk_size=1024):
    with wave.open(path, 'rb') as wf:
        pyaudio_format = pa.get_format_from_width(wf.getsampwidth())
        numpy_dtype = PYAUDIO_TO_NUMPY[pyaudio_format]
        channels = wf.getnchannels()
        rate = wf.getframerate()
        sample_width = wf.getsampwidth()

        parts = []
        data = wf.readframes(chunk_size)
        while data != b'':
            part = np.frombuffer(data, dtype=numpy_dtype)
            parts.append(part)
            data = wf.readframes(chunk_size)

        data = np.concatenate(parts)
        return Sound(pyaudio_format, channels, rate, sample_width, data)


def load_pcm(path, dtype, rate):
    pyaudio_format = NUMPY_TO_PYAUDIO[dtype]
    sample_width = pa.get_sample_size(pyaudio_format)

    data = np.fromfile(path, dtype)

    return Sound(pyaudio_format, 1, rate, sample_width, data)


def save_pcm(path, sound):
    sound.data.tofile(path)
