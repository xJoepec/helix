from __future__ import annotations

import json
import os
import sys
import unittest

import numpy as np

# Ensure the 'code' directory (package root for 'helix') is on sys.path
PKG_ROOT = os.path.dirname(os.path.dirname(__file__))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)


class TestSerialize(unittest.TestCase):
    def test_to_jsonable_numpy(self):
        from helix.serialize import json_dumps, to_jsonable

        data = {
            "arr": np.array([[1, 2], [3, 4]], dtype=np.int32),
            "pi": np.float64(3.14),
            "items": [np.int64(5), np.array([1.0, 2.0])],
        }
        conv = to_jsonable(data)

        # Ensure numpy types are converted
        self.assertEqual(conv["arr"], [[1, 2], [3, 4]])
        self.assertIsInstance(conv["pi"], float)
        self.assertEqual(conv["items"], [5, [1.0, 2.0]])

        # Should be dumpable to JSON
        s = json_dumps(data)
        parsed = json.loads(s)
        self.assertEqual(parsed["arr"], [[1, 2], [3, 4]])


if __name__ == "__main__":
    unittest.main()
