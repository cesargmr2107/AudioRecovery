import numpy
import util
import scipy.io.wavfile as wav
from TruncatedBinaryHamming import TruncatedBinaryHamming

if __name__ == "__main__":
    # Load music file and convert it to numpy uint16 array
    rate, data = wav.read("music_files/original.wav")
    converted = data.astype(dtype=numpy.uint16)

    # Encode audio
    encoded_audio = []
    encoder = TruncatedBinaryHamming(16)
    for sample in converted:
        bin_sample = util.int_to_bin(sample, 16)
        encoded_sample = encoder.encode_source_word(bin_sample)
        encoded_audio.append(encoded_sample)

    # Store encoded_audio audio file
    numpy_encoded_array = numpy.array(encoded_audio)
    numpy.save("music_files/encoded_audio", numpy_encoded_array)
