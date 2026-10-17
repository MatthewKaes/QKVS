from __future__ import annotations

import json

from qkvs.runtime import environment


def main() -> None:
    print(json.dumps(environment(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
