import json
import sys
import tomllib
from pathlib import Path


def load_ruff_excludes(ruff_config_path: Path) -> list[str]:
    try:
        with ruff_config_path.open("rb") as f:
            ruff_config = tomllib.load(f)

        excludes = ruff_config.get("exclude", [])
        if not isinstance(excludes, list):
            print(
                f"Error: exclude patterns in {ruff_config_path} is not a list"
            )
            sys.exit(1)

        return excludes  # noqa: TRY300

    except FileNotFoundError:
        print(f"Error: {ruff_config_path} not found")
        sys.exit(1)
    except (OSError, tomllib.TOMLDecodeError) as e:
        print(f"Error reading {ruff_config_path}: {e}")
        sys.exit(1)


def update_pyright_config(
    pyright_config_path: Path, excludes: list[str]
) -> None:
    try:
        with pyright_config_path.open() as f:
            current_config = json.load(f)

        existing_excludes = current_config.get("exclude", [])

        common_excludes = ["**/node_modules", "**/__pycache__"]
        preserved_excludes = [
            exc for exc in existing_excludes if exc in common_excludes
        ]

        new_excludes = preserved_excludes + excludes

        seen = set()
        deduped_excludes: list[str] = []
        for item in new_excludes:
            if item not in seen:
                seen.add(item)
                deduped_excludes.append(item)

        desired_config = dict(current_config)
        desired_config["exclude"] = deduped_excludes

        # 🔑 Idempotency check
        if desired_config == current_config:
            print("✅ pyrightconfig.json already up to date")
            return

        # Write with trailing newline to satisfy end-of-file-fixer
        with pyright_config_path.open("w") as f:
            json.dump(desired_config, f, indent=2)
            f.write("\n")

        print(
            f"✅ Updated {pyright_config_path} with {len(excludes)} exclude patterns from ruff.toml"
        )

    except FileNotFoundError:
        print(f"Error: {pyright_config_path} not found")
        sys.exit(1)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error updating {pyright_config_path}: {e}")
        sys.exit(1)


def main() -> None:
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    ruff_config_path = project_root / "ruff.toml"
    pyright_config_path = project_root / "pyrightconfig.json"

    print(
        f"🔄 Syncing exclude patterns from {ruff_config_path.name} "
        f"to {pyright_config_path.name}"
    )

    excludes = load_ruff_excludes(ruff_config_path)
    print(f"📋 Found {len(excludes)} exclude patterns in ruff.toml")

    update_pyright_config(pyright_config_path, excludes)


if __name__ == "__main__":
    main()
