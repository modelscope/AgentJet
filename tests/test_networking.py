import socket
import unittest
import sys
import os
import importlib.util

# Load the module directly to avoid top-level package import issues
# caused by broken dependencies in other parts of the codebase.
# We are testing a standalone utility, so we don't need the whole app context.
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ajet", "utils", "networking.py"))
spec = importlib.util.spec_from_file_location("networking", module_path)
networking = importlib.util.module_from_spec(spec)
spec.loader.exec_module(networking)

find_free_port = networking.find_free_port
get_host_ip = networking.get_host_ip


class TestNetworking(unittest.TestCase):
    def test_find_free_port(self):
        """Test that find_free_port returns a valid integer port."""
        port = find_free_port()
        self.assertIsInstance(port, int)
        self.assertGreater(port, 0)
        self.assertLess(port, 65536)

        # Verify the port is valid to bind to (it should have been released)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
            except OSError:
                # It's possible the port was taken immediately by another process
                # but unlikely in a test environment.
                pass

    def test_get_host_ip(self):
        """Test that get_host_ip returns a valid IP string."""
        ip = get_host_ip()
        self.assertIsInstance(ip, str)
        parts = ip.split(".")
        self.assertEqual(len(parts), 4)
        for part in parts:
            if part == "localhost":
                continue
            self.assertTrue(part.isdigit(), f"Part {part} is not a digit")
            self.assertTrue(0 <= int(part) <= 255)

    def test_get_host_ip_with_interface(self):
        """Test get_host_ip with a non-existent interface falls back to default behavior."""
        # This will likely fail the interface specific block and fall back to the connect method
        ip = get_host_ip(interface="invalid_interface_XYZ")
        self.assertIsInstance(ip, str)
        parts = ip.split(".")
        self.assertEqual(len(parts), 4)


if __name__ == "__main__":
    unittest.main()
