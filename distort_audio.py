import numpy
import util
import scipy.io.wavfile as wav

from TruncatedBinaryHamming import TruncatedBinaryHamming

if __name__ == "__main__":
    # Load encoded_audio audio file
    encoded_audio = numpy.load("music_files/encoded_audio.npy")

    # Create encoder to get info positions
    encoder = TruncatedBinaryHamming(wanted_k=16)

    # Generate distorted encoded_audio file and distorted audio
    distorted_encoded = []
    distorted_audio = []
    for encoded_sample in encoded_audio:
        # Generate distorted sample and store it in distorted encoded_audio array
        distorted_encoded_sample = util.gen_random_errors(vector=encoded_sample, n_errors=1)[0]
        distorted_encoded.append(distorted_encoded_sample)
        # Get audio info from info bits, convert it and store in audio
        information_bits = numpy.array([
            distorted_encoded_sample[i] for i in range(len(distorted_encoded_sample)) if encoder.info_positions[i]
        ])
        sample = util.bin_to_int(information_bits)
        distorted_audio.append(sample)

    # Store distorted encoded_audio file
    numpy_encoded_array = numpy.array(distorted_encoded)
    numpy.save("music_files/distorted", numpy_encoded_array)

    # Store distorted audio file
    numpy_audio_array = numpy.array(distorted_audio, dtype=numpy.int16)
    wav.write(filename="music_files/distorted.wav", rate=22050, data=numpy_audio_array)
