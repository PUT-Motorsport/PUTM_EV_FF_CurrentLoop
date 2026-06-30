"""
Check which .mcap files in data/rosbag2 have torque setpoints
only from rear motors (front target_torque always == 0).

AMK message layout (CDR, little-endian):
  [0:4]  CDR header
  [4:8]  AmkControl (4 x bool: inverter_on, dc_on, enable, error_reset)
  [8:10] target_torque     (int16)
  [10:12] torque_pos_limit (int16)
  [12:14] torque_neg_limit (int16)
"""

import struct
from pathlib import Path
from mcap.reader import make_reader

DATA_DIR = Path(__file__).parent.parent / "data" / "rosbag2"

FRONT_TOPICS = [
    "/putm_vcl/amk/front/left/setpoints",
    "/putm_vcl/amk/front/right/setpoints",
]
REAR_TOPICS = [
    "/putm_vcl/amk/rear/left/setpoints",
    "/putm_vcl/amk/rear/right/setpoints",
]
ALL_TOPICS = FRONT_TOPICS + REAR_TOPICS

TARGET_TORQUE_OFFSET = 8  # bytes from start of CDR payload


def decode_target_torque(data: bytes) -> int:
    return struct.unpack_from("<h", data, TARGET_TORQUE_OFFSET)[0]


def analyse_mcap(path: Path) -> dict | None:
    front_nonzero = 0
    rear_nonzero = 0
    front_total = 0
    rear_total = 0

    with open(path, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        available = {ch.topic for ch in summary.channels.values()}
        topics_to_read = [t for t in ALL_TOPICS if t in available]

        if not topics_to_read:
            return None  # no AMK setpoint topics in this file

        for _, channel, message in reader.iter_messages(topics=topics_to_read):
            torque = decode_target_torque(message.data)
            if channel.topic in FRONT_TOPICS:
                front_total += 1
                if torque != 0:
                    front_nonzero += 1
            else:
                rear_total += 1
                if torque != 0:
                    rear_nonzero += 1

    return {
        "front_total": front_total,
        "front_nonzero": front_nonzero,
        "rear_total": rear_total,
        "rear_nonzero": rear_nonzero,
    }


def main():
    mcap_files = sorted(DATA_DIR.glob("*.mcap"))

    if not mcap_files:
        print(f"No .mcap files found in: {DATA_DIR}")
        return

    rear_only = []

    for path in mcap_files:
        print(f"Analysing: {path.name} ...", end=" ", flush=True)
        stats = analyse_mcap(path)

        if stats is None:
            print("no AMK setpoint topics — skipping")
            continue

        has_rear = stats["rear_nonzero"] > 0
        has_front = stats["front_nonzero"] > 0

        if has_rear and not has_front:
            status = "REAR ONLY"
            rear_only.append(path.name)
        elif has_rear and has_front:
            status = "front + rear"
        elif not has_rear and not has_front:
            status = "no torque (all zeros)"
        else:
            status = "front only"

        print(
            f"{status} | "
            f"front nonzero={stats['front_nonzero']}/{stats['front_total']} | "
            f"rear nonzero={stats['rear_nonzero']}/{stats['rear_total']}"
        )

    print()
    print("=" * 60)
    if rear_only:
        print(f"Files with REAR-ONLY setpoints ({len(rear_only)}):")
        for name in rear_only:
            print(f"  {name}")
    else:
        print("No files with exclusively rear-wheel setpoints found.")


if __name__ == "__main__":
    main()
