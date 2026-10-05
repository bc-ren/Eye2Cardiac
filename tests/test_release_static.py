"""Standard-library checks of the public release scanner, using synthetic text."""
import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("release_check", ROOT / "scripts/check_release.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class StaticChecks(unittest.TestCase):
    def test_clean_source(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "example.py").write_text("import json\nfrom pathlib import Path\nx = 2\n")
            result = CHECK.scan(root)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["python_syntax_checked"], 1)

    def test_rejects_private_path_without_echoing_it(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            private = "/" + "home" + "/synthetic_user/private"
            (root / "example.py").write_text("x = " + repr(private))
            result = CHECK.scan(root)
            self.assertIn("private_absolute_path", {x["rule"] for x in result["findings"]})
            self.assertNotIn(private, str(result))

    def test_rejects_binary_artifacts(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "forbidden.pt").write_bytes(b"synthetic; not a checkpoint")
            self.assertEqual(CHECK.scan(root)["status"], "FAIL")

    def test_rejects_unknown_import_and_bad_syntax(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "unknown.py").write_text("import absent_dependency_for_test\n")
            (root / "broken.py").write_text("def broken(:\n")
            rules = {x["rule"] for x in CHECK.scan(root)["findings"]}
            self.assertTrue({"unresolved_import", "python_syntax"} <= rules)


if __name__ == "__main__":
    unittest.main()
