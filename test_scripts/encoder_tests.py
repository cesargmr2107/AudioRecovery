from TruncatedBinaryHamming import TruncatedBinaryHamming

if __name__ == "__main__":

    # Create Hamming encoder
    encoder = TruncatedBinaryHamming(wanted_k=16)

    # Show parameters and parity matrix
    encoder.show_parameters()
    encoder.show_parity_matrix()
    encoder.show_generator_matrix()
    encoder.show_information_positions()

    # Do tests
    source_words = [
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0]
    ]
    for word in source_words:
        encoder.test(word, 2, 2)
