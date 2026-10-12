"""Control-flow regression tests; scheduling bits do not create CFG edges."""
import unittest

from ..sass import cfg


def rows(*instructions):
    return [{"pc": i * 16, "sass": ins, "control": 0}
            for i, ins in enumerate(instructions)]


class CFGTests(unittest.TestCase):
    def test_unconditional_traps_terminate_and_split(self):
        for instruction in ("BPT.TRAP 0x1;", "TRAP;"):
            with self.subTest(instruction=instruction):
                graph = cfg(rows(instruction, "MOV R0, R1;", "EXIT;"))
                self.assertTrue(graph["complete"])
                self.assertEqual([b["start"] for b in graph["blocks"]], [0, 16])
                self.assertEqual(graph["blocks"][0]["successors"], [])
                self.assertFalse(graph["blocks"][1]["reachable"])

    def test_predicated_traps_keep_only_non_taken_fallthrough(self):
        for predicate in ("@P0", "@!P0", "@UP1", "@!UP1"):
            for instruction in ("BPT.TRAP 0x1;", "TRAP;"):
                with self.subTest(predicate=predicate, instruction=instruction):
                    graph = cfg(rows(f"{predicate} {instruction}", "EXIT;"))
                    self.assertTrue(graph["complete"])
                    self.assertEqual(graph["blocks"][0]["successors"], [16])
                    self.assertTrue(graph["blocks"][1]["reachable"])

    def test_unknown_breakpoint_forms_are_not_assumed_to_trap(self):
        for instruction in ("BPT 0x1;", "BPT.BREAK 0x1;", "BPT.DRAIN 0x1;",
                            "BPT.TRAP.UNKNOWN 0x1;"):
            with self.subTest(instruction=instruction):
                graph = cfg(rows(instruction, "EXIT;"))
                self.assertFalse(graph["complete"])
                self.assertEqual(graph["errors"], [])
                self.assertEqual(len(graph["unknowns"]), 1)
                self.assertIn(instruction.split()[0], graph["unknowns"][0])
                # Possible fallthrough is retained, but the graph is explicitly
                # incomplete because debugger/breakpoint behavior is unknown.
                self.assertEqual(graph["blocks"][0]["successors"], [16])
                self.assertTrue(graph["blocks"][1]["reachable"])

    def test_branch_can_skip_a_trap(self):
        graph = cfg(rows("@P0 BRA 0x20;", "BPT.TRAP 0x1;", "EXIT;"))
        self.assertTrue(graph["complete"])
        self.assertEqual([b["successors"] for b in graph["blocks"]],
                         [[16, 32], [], []])
        self.assertTrue(all(b["reachable"] for b in graph["blocks"]))

    def test_final_trap_has_no_fallthrough(self):
        graph = cfg(rows("MOV R0, R1;", "BPT.TRAP 0x1;"))
        self.assertTrue(graph["complete"])
        self.assertEqual(len(graph["blocks"]), 1)
        self.assertEqual(graph["blocks"][0]["successors"], [])

    def test_predicated_arithmetic_does_not_split_blocks(self):
        graph = cfg(rows("@P0 MOV R0, R1;", "IADD3 R0, R0, 0x1, RZ;", "EXIT;"))
        self.assertTrue(graph["complete"])
        self.assertEqual(len(graph["blocks"]), 1)
        self.assertEqual(graph["blocks"][0]["instructions"], 3)


if __name__ == "__main__":
    unittest.main()
