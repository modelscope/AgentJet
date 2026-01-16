import socket


def find_free_port() -> int:
    """Find a free port on the system."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def get_host_ip(interface=None):
    """
    get the host machine's IP address.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if interface:
            try:
                import fcntl
                import struct

                ip = socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x8915, struct.pack("256s", interface[:15].encode()))[20:24])  # SIOCGIFADDR
                return ip
            except (ImportError, IOError):
                pass
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip

    except Exception:
        return "127.0.0.1"
