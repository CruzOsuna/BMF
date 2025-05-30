import os

def verify_rcpnl(base_directory):
    print(f"Checking .rcpnl files in: {base_directory}\n")

    errors = []
    subfolder_summary = {}  # Dictionary: subfolder -> number of .rcpnl files
    total_subfolders = 0

    for root, dirs, files in os.walk(base_directory):
        rcpnl_files = [f for f in files if f.endswith('.rcpnl')]

        if not rcpnl_files:
            continue

        subfolder_name = os.path.basename(root)
        total_subfolders += 1
        subfolder_summary[subfolder_name] = len(rcpnl_files)

        print(f"📁 Subfolder: {subfolder_name} ({len(rcpnl_files)} files)")
        for file in sorted(rcpnl_files):
            filepath = os.path.join(root, file)
            try:
                filesize = os.path.getsize(filepath)
                if filesize == 0:
                    errors.append(f"⚠️ Empty file: {filepath}")
                    print(f"  ⚠️ Empty → {file}")
                else:
                    print(f"  ✅ OK    → {file} ({filesize} bytes)")
            except Exception as e:
                errors.append(f"❌ Error reading {filepath}: {str(e)}")
                print(f"  ❌ Error → {file} ({str(e)})")

        print()

    # Show errors
    if errors:
        print("\n=== ERRORS DETECTED ===")
        for err in errors:
            print(err)
    else:
        print("\n✅ All .rcpnl files are in good condition.")

    # Show summary
    print("\n=== OVERALL SUMMARY ===")
    print(f"📂 Total subfolders with .rcpnl files: {total_subfolders}")
    for subfolder, count in sorted(subfolder_summary.items()):
        print(f"  - {subfolder}: {count} .rcpnl files")

if __name__ == "__main__":
    rcpnl_directory = "/media/cruz/Spatial/t-CycIF_human_2025_2/01_Registration/RCPNL"
    verify_rcpnl(rcpnl_directory)
