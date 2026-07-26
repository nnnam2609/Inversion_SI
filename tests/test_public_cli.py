from __future__ import annotations

import contextlib
import io
import unittest

from src import cli
from src.orchestration import grid_transform


class PublicCliTest(unittest.TestCase):
    def test_root_help_lists_the_single_command_tree(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = cli.main(["--help"])
        self.assertEqual(status, 0)
        self.assertIn("preprocess sessions", output.getvalue())
        self.assertIn("adapt", output.getvalue())

    def test_grid_transform_help_does_not_resolve_help_as_a_script(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = grid_transform.main(["--help"])
        self.assertEqual(status, 0)
        self.assertIn("grid-transform", output.getvalue())


if __name__ == "__main__":
    unittest.main()
