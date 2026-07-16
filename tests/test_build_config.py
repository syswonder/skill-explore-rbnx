from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class BuildConfigTest(unittest.TestCase):
    def test_build_uses_reproducible_python_codegen(self) -> None:
        build = (ROOT / "scripts" / "build.sh").read_text()
        self.assertIn("scripts/run_python_codegen.sh", build)

    def test_codegen_and_runtime_versions_match(self) -> None:
        helper = (ROOT / "scripts" / "run_python_codegen.sh").read_text()
        dockerfile = (ROOT / "docker" / "Dockerfile").read_text()
        self.assertIn('PROTOBUF_VERSION="6.33.6"', helper)
        self.assertIn('GRPCIO_VERSION="1.80.0"', helper)
        self.assertIn("protobuf==6.33.6", dockerfile)
        self.assertIn("grpcio==1.80.0", dockerfile)

    def test_codegen_import_smoke_covers_atlas_and_skill_stubs(self) -> None:
        helper = (ROOT / "scripts" / "run_python_codegen.sh").read_text()
        self.assertIn("import atlas_pb2_grpc", helper)
        self.assertIn("import explore_pb2", helper)
        self.assertIn("import explore_mcp", helper)
        self.assertIn("import robonix_contracts_pb2_grpc", helper)


if __name__ == "__main__":
    unittest.main()
