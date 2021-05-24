import numpy
import util
import scipy.io.wavfile as wav
from TruncatedBinaryHamming import TruncatedBinaryHamming

if __name__ == "__main__":

    # Load encoded_audio audio file
    encoded_audio = numpy.load("../music_files/distorted.npy")

    # Decode audio
    decoded_audio = []
    encoder = TruncatedBinaryHamming(16)
    for encoded_sample in encoded_audio:
        decoded_sample = encoder.decode_word(encoded_sample)[0]
        sample = util.bin_to_int(decoded_sample)
        decoded_audio.append(sample)

    # Store decoded_audio audio file
    numpy_decoded_array = numpy.array(decoded_audio, dtype='int16')
    wav.write(filename="../music_files/distorted_recovered.wav", rate=22050, data=numpy_decoded_array)
