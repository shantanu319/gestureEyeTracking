from __future__ import annotations

import unittest

from pointer_filter import AdaptivePointerFilter


class AdaptivePointerFilterTest(unittest.TestCase):
    def test_small_jitter_stays_in_deadzone(self) -> None:
        filt = AdaptivePointerFilter((1920, 1080))
        self.assertEqual(filt.update((400, 300)), (400, 300))
        self.assertEqual(filt.update((402, 301)), (400, 300))
        self.assertEqual(filt.update((403, 299)), (400, 300))

    def test_large_movement_adapts_quickly(self) -> None:
        filt = AdaptivePointerFilter((1920, 1080))
        filt.update((100, 100))
        moved = filt.update((900, 600))
        self.assertIsNotNone(moved)
        self.assertGreater(moved[0], 600)
        self.assertGreater(moved[1], 400)

    def test_missing_frames_hold_then_clear(self) -> None:
        filt = AdaptivePointerFilter((1920, 1080), hold_frames=2)
        filt.update((200, 100))
        self.assertEqual(filt.update(None), (200, 100))
        self.assertEqual(filt.update(None), (200, 100))
        self.assertIsNone(filt.update(None))


if __name__ == "__main__":
    unittest.main()
