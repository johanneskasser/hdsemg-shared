from hdsemg_shared.fileio.file_io import EMGFile

def main():
    emg = EMGFile.load("C:\\Users\\johan\\hdsemg\\data\\random\\P_H_CE20250320132736_08.mat")

    print(f"Data shape: {emg.data.shape}")
    print(f"Time shape: {emg.time.shape}")
    print(f"Description: {emg.description}")
    print(f"Sampling Frequency: {emg.sampling_frequency}")
    print(f"File Name: {emg.file_name}")
    print(f"File Size: {emg.file_size}")
    print(f"File Type: {emg.file_type}")
    print(f"Grids: {emg.grids}")
    print(f"Channel Count: {emg.channel_count}")

    refs = _get_all_reference_indices(emg)
    print(f"Reference Indices: {refs}")


def _get_all_reference_indices(emg) -> list[int]:
    """
    Retrieves all reference-signal indices from global grid info.
    """
    refs: list[int] = []
    for grid in emg.grids:
        for ref in grid.ref_indices:
            if isinstance(ref, int):
                refs.append(ref)
    return refs


if __name__ == "__main__":
    main()