def decode_test(encoded_word, error_vector=None):
    decoded, syndrome, error = encoder.decode_word(encoded_word)
    print(f"Decoding {encoded_word}:")
    print(f"\t- Detected error in {encoded_word}? {error}")
    print(f"\t- Syndrome: {syndrome}")
    if error:
        print(f"\t- Error vector: {error_vector}")
    print(f"\t- Decoded word: {decoded}\n")


def test(source_word, encoder, n_words, n_errors_in_word):
    print(f"TEST\n{'=' * 50}")
    encoded_word = encoder.encode_source_word(source_word)
    print(f"Source word: {source_word}\n"
          f"Encoded word: {encoded_word}\n")
    decode_test(encoded_word)
    for i in range(n_words):
        bad_encoded_word, error_vector = util.gen_random_errors(encoded_word, n_errors_in_word)
        decode_test(bad_encoded_word, error_vector)