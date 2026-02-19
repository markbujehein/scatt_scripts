# Root-level non-Mantid test package.
#
# Tests here are designed to run without the Mantid framework by using
# inline replicas or the mock_mantid stub (tests/mock_mantid.py).
#
# Run with:
#   python -m pytest tests/ -v -k "not mantid"
# or:
#   python -m unittest discover -s tests -p 'test_*.py'
