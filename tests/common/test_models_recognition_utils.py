import numpy as np
import pytest

from doctr.models.recognition.core import aggregate_confidence
from doctr.models.recognition.utils import merge_multi_strings, merge_strings


@pytest.mark.parametrize(
    "a, b, overlap_ratio, merged",
    [
        # Last character of first string and first of last string will be cropped when merging - indicated by X
        ("abcX", "Xdef", 0.5, "abcdef"),
        ("abcdX", "Xdef", 0.75, "abcdef"),
        ("abcdeX", "Xdef", 0.9, "abcdef"),
        ("abcdefX", "Xdef", 0.9, "abcdef"),
        # Long repetition - four of seven characters in the second string are in the estimated overlap
        # X-chars will be cropped during merge, because they might be cut off during splitting of corresponding image
        ("abccccX", "Xcccccc", 4 / 7, "abcccccccc"),
        ("abc", "", 0.5, "abc"),
        ("", "abc", 0.5, "abc"),
        ("a", "b", 0.5, "ab"),
        # No overlap of input strings after crop
        ("abcdX", "Xefghi", 0.33, "abcdefghi"),
        # No overlap of input strings after crop with shorter inputs
        ("bcdX", "Xefgh", 0.4, "bcdefgh"),
        # No overlap of input strings after crop with even shorter inputs
        ("cdX", "Xefg", 0.5, "cdefg"),
        # Full overlap of input strings
        ("abcdX", "Xbcde", 1.0, "abcde"),
        # One repetition within inputs
        ("ababX", "Xabde", 0.8, "ababde"),
        # Multiple repetitions within inputs
        ("ababX", "Xabab", 0.8, "ababab"),
        # Multiple repetitions within inputs with shorter input strings
        ("abaX", "Xbab", 1.0, "abab"),
        # Longer multiple repetitions within inputs with half overlap
        ("cabababX", "Xabababc", 0.5, "cabababababc"),
        # Longer multiple repetitions within inputs with full overlap
        ("ababaX", "Xbabab", 1.0, "ababab"),
        # One different letter in overlap
        ("one_differon", "ferent_letter", 0.5, "one_differont_letter"),
        # First string empty after crop
        ("-", "test", 0.9, "-test"),
        # Second string empty after crop
        ("test", "-", 0.9, "test-"),
    ],
)
def test_merge_strings(a, b, overlap_ratio, merged):
    assert merged == merge_strings(a, b, overlap_ratio)


@pytest.mark.parametrize(
    "seq_list, overlap_ratio, last_overlap_ratio, merged",
    [
        # One character at each conjunction point will be cropped when merging - indicated by X
        (["abcX", "Xdef"], 0.5, 0.5, "abcdef"),
        (["abcdX", "XdefX", "XefghX", "Xijk"], 0.5, 0.5, "abcdefghijk"),
        (["abcdX", "XdefX", "XefghiX", "Xaijk"], 0.5, 0.8, "abcdefghijk"),
        (["aaaa", "aaab", "aabc"], 0.8, 0.3, "aaaabc"),
        # Handle empty input
        ([], 0.5, 0.4, ""),
    ],
)
def test_merge_multi_strings(seq_list, overlap_ratio, last_overlap_ratio, merged):
    assert merged == merge_multi_strings(seq_list, overlap_ratio, last_overlap_ratio)


# Tests for confidence aggregation
class TestAggregateConfidence:
    """Tests for the aggregate_confidence function."""

    def test_empty_probs(self):
        """Empty probability array should return 0.0."""
        assert aggregate_confidence(np.array([]), "mean") == 0.0
        assert aggregate_confidence([], "min") == 0.0

    @pytest.mark.parametrize(
        "probs, method, expected",
        [
            # Arithmetic mean tests
            ([0.8, 0.9, 0.7], "mean", 0.8),
            ([0.5, 0.5, 0.5], "mean", 0.5),
            ([1.0, 1.0, 1.0], "mean", 1.0),
            ([0.0, 0.0, 0.0], "mean", 0.0),
            # Minimum tests
            ([0.8, 0.9, 0.7], "min", 0.7),
            ([0.5, 0.3, 0.9], "min", 0.3),
            ([1.0, 1.0, 1.0], "min", 1.0),
            # Maximum tests
            ([0.8, 0.9, 0.7], "max", 0.9),
            ([0.5, 0.3, 0.9], "max", 0.9),
            ([0.0, 0.0, 0.0], "max", 0.0),
        ],
    )
    def test_basic_aggregation_methods(self, probs, method, expected):
        """Test basic aggregation methods with simple inputs."""
        result = aggregate_confidence(probs, method)
        assert abs(result - expected) < 1e-6

    def test_geometric_mean(self):
        """Test geometric mean calculation."""
        # geometric_mean([0.8, 0.8, 0.8]) = 0.8
        result = aggregate_confidence([0.8, 0.8, 0.8], "geometric_mean")
        assert abs(result - 0.8) < 1e-6

        # geometric_mean([1.0, 0.5]) = sqrt(0.5) ≈ 0.707
        result = aggregate_confidence([1.0, 0.5], "geometric_mean")
        assert abs(result - np.sqrt(0.5)) < 1e-6

        # geometric_mean with a zero should return very small value (using epsilon)
        result = aggregate_confidence([0.0, 0.5, 0.5], "geometric_mean")
        assert result < 0.01  # Should be very small due to zero

    def test_harmonic_mean(self):
        """Test harmonic mean calculation."""
        # harmonic_mean([0.5, 0.5, 0.5]) = 0.5
        result = aggregate_confidence([0.5, 0.5, 0.5], "harmonic_mean")
        assert abs(result - 0.5) < 1e-6

        # harmonic_mean([1.0, 0.5]) = 2 / (1/1.0 + 1/0.5) = 2 / 3 ≈ 0.667
        result = aggregate_confidence([1.0, 0.5], "harmonic_mean")
        assert abs(result - 2 / 3) < 1e-6

        # harmonic_mean with a zero should return very small value (using epsilon)
        result = aggregate_confidence([0.0, 0.5, 0.5], "harmonic_mean")
        assert result < 0.01  # Should be very small due to zero

    def test_clipping(self):
        """Test that values are clipped to [0, 1] range."""
        # Values outside range should be clipped
        result = aggregate_confidence([1.5, 0.5, -0.5], "mean")
        # After clipping: [1.0, 0.5, 0.0], mean = 0.5
        assert abs(result - 0.5) < 1e-6

    def test_single_value(self):
        """Test with single value - all methods should return that value."""
        for method in ["mean", "geometric_mean", "harmonic_mean", "min", "max"]:
            result = aggregate_confidence([0.75], method)
            assert abs(result - 0.75) < 1e-6

    def test_custom_callable(self):
        """Test with custom aggregation function."""

        def custom_median(probs):
            return float(np.median(probs))

        result = aggregate_confidence([0.1, 0.5, 0.9], custom_median)
        assert abs(result - 0.5) < 1e-6

    def test_invalid_method(self):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown aggregation method"):
            aggregate_confidence([0.5, 0.5], "invalid_method")

    def test_numpy_array_input(self):
        """Test with numpy array input."""
        probs = np.array([0.8, 0.9, 0.7])
        result = aggregate_confidence(probs, "mean")
        assert abs(result - 0.8) < 1e-6

    def test_list_input(self):
        """Test with Python list input."""
        probs = [0.8, 0.9, 0.7]
        result = aggregate_confidence(probs, "mean")
        assert abs(result - 0.8) < 1e-6

    def test_ordering_sensitivity(self):
        """Test that methods sensitive to outliers behave correctly."""
        # Low outlier should affect min and harmonic_mean more than mean
        probs_with_low_outlier = [0.9, 0.9, 0.9, 0.1]

        mean_result = aggregate_confidence(probs_with_low_outlier, "mean")
        min_result = aggregate_confidence(probs_with_low_outlier, "min")
        harmonic_result = aggregate_confidence(probs_with_low_outlier, "harmonic_mean")

        # min should return the lowest value
        assert abs(min_result - 0.1) < 1e-6
        # mean should be higher
        assert mean_result > harmonic_result
        # harmonic mean should be more affected by low values than arithmetic mean
        assert harmonic_result < mean_result
