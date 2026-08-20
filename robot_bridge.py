import socket
import struct
import threading
import time
import json
import app_config


def _encode_str(s: str) -> bytes:
    b = s.encode()
    return struct.pack("!H", len(b)) + b


def _encode_varint(n: int) -> bytes:
    out = bytearray()
    while True:
        d = n % 128
        n //= 128
        if n > 0:
            d |= 0x80
        out.append(d)
        if n == 0:
            break
    return bytes(out)


def _send(sock, header: int, payload: bytes):
    rl = _encode_varint(len(payload))
    sock.sendall(bytes([header]) + rl + payload)


def _recv_exact(sock, n: int):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _recv_packet(sock):
    try:
        b1 = sock.recv(1)
        if not b1:
            return None

        mult, val = 1, 0
        while True:
            b = sock.recv(1)
            if not b:
                return None
            enc = b[0]
            val += (enc & 127) * mult
            if (enc & 128) == 0:
                break
            mult *= 128

        payload = _recv_exact(sock, val)
        if payload is None:
            return None

        return (b1[0] >> 4, b1[0] & 0x0F, payload)
    except Exception:
        return None


def send_robot_step(step_id: int) -> bool:
    """Fresh TCP connection per publish — avoids stale socket / silent drop issues."""
    for attempt in range(3):
        s = None
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5.0)
            s.connect((app_config.MQTT_BROKER, app_config.MQTT_PORT))

            # MQTT CONNECT
            client_id = f"robot-pub-{int(time.time())}"
            connect_payload = (
                _encode_str("MQTT") +
                b"\x04" +                    # protocol level MQTT 3.1.1
                b"\x02" +                    # connect flags: clean session
                struct.pack("!H", 60) +      # keepalive 60 s
                _encode_str(client_id)
            )
            _send(s, 0x10, connect_payload)

            connack = _recv_exact(s, 4)
            if not connack or len(connack) < 4:
                raise RuntimeError("CONNACK incomplet")
            if connack[3] != 0:
                raise RuntimeError(f"MQTT CONNECT refuzat, cod={connack[3]}")

            # MQTT PUBLISH QoS 0
            pub_payload = _encode_str(app_config.MQTT_TOPIC_STEP) + str(step_id).encode()
            _send(s, 0x30, pub_payload)

            # Allow the OS to flush the send buffer before closing
            time.sleep(0.15)
            s.close()

            print(f"[MQTT] ✓ Trimis pasul {step_id} pe {app_config.MQTT_TOPIC_STEP}")
            return True

        except Exception as e:
            print(f"[MQTT] Eroare publish (attempt {attempt + 1}/3): {e}")
            if s:
                try:
                    s.close()
                except Exception:
                    pass

    return False


class RobotStatusListener:
    def __init__(self, on_done=None, on_error=None):
        self.on_done = on_done
        self.on_error = on_error
        self._thread = None
        self._stop_event = threading.Event()
        self._sock = None

    def _connect_local(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5.0)
        s.connect((app_config.MQTT_BROKER, app_config.MQTT_PORT))

        payload = (
            _encode_str("MQTT") +
            b"\x04" +
            b"\x02" +
            struct.pack("!H", 30) +
            _encode_str(f"status-listener-{int(time.time())}")
        )
        _send(s, 0x10, payload)

        connack = s.recv(4)
        if not connack:
            raise RuntimeError("MQTT broker nu a răspuns la CONNECT pentru listener")

        self._sock = s
        return s

    def _subscribe(self, topic: str, packet_id: int):
        pid = struct.pack("!H", packet_id)
        payload = pid + _encode_str(topic) + bytes([0])
        _send(self._sock, 0x82, payload)

    def _parse_publish(self, payload: bytes):
        if not payload or len(payload) < 2:
            return "", ""

        tlen = struct.unpack("!H", payload[0:2])[0]
        topic = payload[2:2 + tlen].decode(errors="replace")
        msg = payload[2 + tlen:]
        return topic, msg.decode(errors="replace")

    def _safe_json(self, text: str):
        try:
            return json.loads(text)
        except Exception:
            t = text.strip()
            if t.isdigit():
                return {"step_id": int(t)}
            return {"raw": t}

    def _run(self):
        while not self._stop_event.is_set():
            try:
                sock = self._connect_local()
                self._subscribe(app_config.MQTT_TOPIC_STATUS_DONE, 1)
                self._subscribe(app_config.MQTT_TOPIC_STATUS_ERROR, 2)
                print("[MQTT] Conectat — ascult statusul robotului...")
                sock.settimeout(0.2)

                while not self._stop_event.is_set():
                    try:
                        pkt = _recv_packet(sock)
                        if pkt is None:
                            continue

                        ptype, _flags, payload = pkt

                        if ptype == 9:  # PINGRESP
                            continue
                        if ptype != 3:  # only PUBLISH
                            continue

                        topic, message = self._parse_publish(payload)
                        data = self._safe_json(message)

                        print(f"[MQTT STATUS] {topic} -> {data}")

                        if topic == app_config.MQTT_TOPIC_STATUS_DONE and self.on_done:
                            self.on_done(data)
                        elif topic == app_config.MQTT_TOPIC_STATUS_ERROR and self.on_error:
                            self.on_error(data)

                    except socket.timeout:
                        continue
                    except Exception as e:
                        print("[MQTT STATUS] Eroare listener:", e)
                        break  # reconnect

            except Exception as e:
                if not self._stop_event.is_set():
                    print(f"[MQTT STATUS] Reconectare în 5s... ({e})")
                    time.sleep(5)
            finally:
                try:
                    if self._sock:
                        self._sock.close()
                        self._sock = None
                except Exception:
                    pass

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        try:
            if self._sock is not None:
                self._sock.close()
        except Exception:
            pass

        if self._thread:
            self._thread.join(timeout=1.5)