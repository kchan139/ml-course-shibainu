import os
import base64
import platform
from pathlib import Path

def main():
    print('Paste the base64-encoded string below:')
    
    b64_input = []
    while True:
        line = input()
        if line.strip() == '':
            break
        b64_input.append(line.strip())

    b64_str = ''.join(b64_input)

    try:
        key_data = base64.b64decode(b64_str)
    except Exception as e:
        print('Error decoding key data: {e}')
        return
    
    secrets_path = Path('secrets')
    secrets_path.mkdir(exist_ok=True)

    key_path = secrets_path / 'key.json'
    with open(key_path, 'wb') as file:
        file.write(key_data)

    absolute_path = key_path.resolve()
    print(f'Key saved to {absolute_path}')

    system = platform.system()
    if system == "Windows":
        print("To set the env var for this session in Command Prompt run:")
        print(f"    set GOOGLE_APPLICATION_CREDENTIALS={absolute_path}")
        print("\nOr in PowerShell:")
        print(f"    $Env:GOOGLE_APPLICATION_CREDENTIALS = \"{absolute_path}\"")
    else:
        # covers Linux, macOS, WSL, etc.
        print("To set the env var for this session in your shell run:")
        print(f"    export GOOGLE_APPLICATION_CREDENTIALS=\"{absolute_path}\"")
    
    print("\nNow you can run your DVC commands")


if __name__ == '__main__':
    main()