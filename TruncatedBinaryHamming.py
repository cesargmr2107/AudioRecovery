import math
import numpy
import util


class TruncatedBinaryHamming:

    def __init__(self, wanted_k):
        # Calculate parameters
        self.k = wanted_k
        self.m = TruncatedBinaryHamming.find_m(wanted_k)
        self.n = self.m + self.k
        # Generate parity and generator matrices
        self.parity_matrix = TruncatedBinaryHamming.calc_parity_matrix(self.m, self.k)
        self.generator_matrix, self.info_positions = TruncatedBinaryHamming.calc_generator_matrix(
            self.parity_matrix, self.k, self.m
        )

    def encode_source_word(self, source_word):
        """
        This method encodes the passed source word with the used Hamming code
        :param source_word:
        :return: the encoded_audio word
        """
        encoded = numpy.matmul(source_word, self.generator_matrix)
        return util.fix_binary_numpy_array(encoded)

    def decode_word(self, encoded_word):
        """
        Given an encoded_audio word, this method will decode it and correct any detected errors
        :param encoded_word: the encoded_audio word to be decoded_audio
        :return: the decoded_audio source word
        """

        # Correct if possible
        syndrome = self.calc_syndrome(encoded_word)
        error = False
        error_position = util.bin_to_int(syndrome)
        if error_position > self.k + self.m:
            return None, syndrome, True
        elif error_position != 0:
            error = True
            error_position -= 1
            encoded_word[error_position] = 1 if encoded_word[error_position] == 0 else 0

        # Decode
        decoded = numpy.array([encoded_word[i] for i in range(len(encoded_word)) if self.info_positions[i]])

        return decoded, syndrome, error

    def calc_syndrome(self, vector: numpy.ndarray):
        """
        This method calculates the syndrome of a vector
        :param vector: the numpy vector
        :return: the calculated syndrome
        """
        pc_transposed = numpy.transpose(self.parity_matrix)
        return util.fix_binary_numpy_array(numpy.matmul(vector, pc_transposed))

    def show_parameters(self):
        """This methods shows the code parameters: k, wanted_k and m"""
        print(
            f"{'=' * 50}\n"
            "PARAMETERS\n"
            f"{'=' * 50}\n"
            f"Number of source digits (wanted_k): {self.k}\n"
            f"Hamming's m parameter: {self.m}\n"
            f"Number of truncated code digits (n = m + k): {self.n}\n"
        )

    def show_parity_matrix(self):
        """This method prints the parity matrix to stdout"""
        print(
            f"{'=' * 50}\n"
            "PARITY MATRIX\n"
            f"{'=' * 50}\n"
        )
        util.show_numpy_matrix(self.parity_matrix)

    def show_generator_matrix(self):
        """This method prints the parity matrix to stdout"""
        print(
            f"{'=' * 50}\n"
            "GENERATOR MATRIX\n"
            f"{'=' * 50}\n"
        )
        util.show_numpy_matrix(self.generator_matrix)

    def show_information_positions(self):
        """This method prints the information positions to stdout"""
        print(
            f"{'=' * 50}\n"
            "INFORMATION POSITIONS\n"
            f"{'=' * 50}"
        )
        print(self.info_positions)

    def test(self, source_word, n_tests, n_errors_in_word):
        """
        Do verbose tests by decoding the encoded_audio source word with k number of errors
        :param source_word: the source word to be encoded_audio
        :param n_tests: number of times to generate errors
        :param n_errors_in_word: the number of errors to be generated in each code word
        """
        print(f"{'=' * 50}\nTESTING {source_word}\n{'=' * 50}")
        encoded_word = self.encode_source_word(source_word)
        print(f"Encoded word: {encoded_word}")
        self.decode_test(encoded_word)
        for n in range(n_tests):
            bad_encoded_word, error_vector = util.gen_random_errors(encoded_word, n_errors_in_word)
            self.decode_test(bad_encoded_word, error_vector)

    def decode_test(self, encoded_word, error_vector=None):
        """
        Verbose test for decoding a word
        :param encoded_word: the word to be decoded_audio
        :param error_vector: the error vector that was applied to generate the error, if any
        """
        original_word = numpy.copy(encoded_word)
        decoded, syndrome, error = self.decode_word(encoded_word)
        print(f"{'-' * 50}")
        print(f"Decoding {encoded_word}:")
        print(f"\t- Detected error in {original_word}? {error}")
        print(f"\t- Syndrome: {syndrome}")
        if error:
            print(f"\t- Error vector: {error_vector}")
        print(f"\t- Decoded word: {decoded}")

    @staticmethod
    def find_m(k):
        """
        This method calculates the m parameter from the wanted_k parameter
        :param k: the wanted_k parameter (number of source digits)
        :return: m, the calculated Hamming's m parameter
        """
        m = 1
        while 2 ** m - m - 1 < k:
            m += 1
        return m

    @staticmethod
    def calc_parity_matrix(m, k):
        """
        This method generates the binary Hamming's parity matrix for the m and k parameters
        :param m: Hamming's m parameter
        :param k: number of code word digits
        :return: the generated parity matrix
        """
        matrix = numpy.zeros(shape=(m, k + m))
        for col_index in range(k + m):
            matrix[:, col_index] = util.int_to_bin(number=col_index + 1, n_bits=m)
        return numpy.array(matrix, dtype='int')

    @staticmethod
    def calc_generator_matrix(parity_matrix, k, m):
        """
        Calculate the Hamming generator matrix in its standard form by using the parity matrix and code parameters
        :param parity_matrix: the Hamming parity matrix
        :param k: the k parameter of the code (number of code word digits)
        :param m: Hamming's m parameter
        :return: the generated parity matrix
        """

        systemized, transposed_q, changes = util.systemize_matrix(parity_matrix)

        # Get standard generator matrix:
        standard_gen_matrix = transposed_q
        for col_index in reversed(range(k)):
            identity_col = util.int_to_bin(2 ** col_index, k)
            standard_gen_matrix = numpy.c_[standard_gen_matrix, identity_col]

        # Revert changes and calculate information positions
        info_positions = [False] * m
        info_positions.extend([True] * k)
        gen_matrix = numpy.copy(standard_gen_matrix)
        for (i, j) in changes:
            standard_col_value = util.bin_to_int(standard_gen_matrix[:, j])
            if math.log(standard_col_value, 2).is_integer():
                info_positions[i] = True
                info_positions[j] = False
            gen_matrix[:, i] = standard_gen_matrix[:, j]

        return gen_matrix.astype('int'), info_positions

