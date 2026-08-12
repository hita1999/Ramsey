import importlib
import sys
import unittest

CORE_SEARCH_MODULES = (
    "book_c1_multi",
    "book_c1c2c3_multi",
    "book_c1c2c3_asymmetric",
    "book_c1toc6_multi",
    "book_c1toc6_asymmetric",
    "book_c1toc10_multi",
)


class PythonCompatibilityTests(unittest.TestCase):
    def test_running_on_supported_python(self) -> None:
        self.assertGreaterEqual(sys.version_info, (3, 13))

    def test_core_search_modules_import(self) -> None:
        for module_name in CORE_SEARCH_MODULES:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)


if __name__ == "__main__":
    unittest.main()
