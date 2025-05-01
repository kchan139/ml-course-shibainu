import subprocess
import sys

dvc_paths = [
    "dataset/processed",
    "dataset/raw",
    "models/experiments",
    "models/trained_models",
]

def main():
    for path in dvc_paths:
        try:
            subprocess.run(['dvc', 'add', path], check=True)
        except subprocess.CalledProcessError as e:
            print(f'Failed to add {path}: return code {e.returncode}', file=sys.stderr)
            sys.exit(e.returncode)

    print('All paths added successfully.')

if __name__ == '__main__':
    main()