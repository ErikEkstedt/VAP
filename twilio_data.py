from glob import glob
import json
import os
import pandas as pd
import tqdm

from vap.data.sliding_window import sliding_window, isvalid_vad_list
from vap.utils.utils import read_json, write_txt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a CSV file with VAP information."
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        help="Path to the directory containing the audio files",
        required=True
    )
    parser.add_argument(
        "--vap_json_path",
        type=str,
        help="Path to the directory containing the JSON files",
        required=True
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        help="Path to the output CSV file",
        default="twilio.csv",
    )

    args = parser.parse_args()

    print("#" * 80)
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("#" * 80)

    # Find vap-json files
    json_files = glob(args.vap_json_path + "/*.json")
    # print("json_files", len(json_files))

    # Create a new list that will contain all new rows
    all_results = []

    bad_json = []

    for json_file in tqdm.tqdm(json_files, desc="Create CSV"):

        name = os.path.basename(json_file).replace(".json", "")
        audio_path = os.path.join(args.audio_path, name + ".wav")

        if os.path.isfile(audio_path):
            # Get the sliding windows

            vad_list = read_json(json_file)
            if not isvalid_vad_list(vad_list):
                bad_json.append(json_file)
                continue

            windows = sliding_window(vad_list)

            # Loop over the results and append each as a new row to the all_results list
            for win in windows:

                # * audio_path
                #     - `PATH/TO/AUDIO.wav`
                # * start: float, definining the start time of the sample
                #     - `0.0`
                # * end: float, definining the end time of the sample
                #     - `20.0`
                # * session: str, the name of the sample session
                #     - `4637`
                # * dataset: str, the name of the dataset
                #     - `twilio`
                # * vad_list: a list containing the voice-activity start/end-times inside of the `start`/`end` times of the row-sample
                all_results.append(
                    {
                        "audio_path": audio_path,
                        "start": win["start"],
                        "end": win["end"],
                        "dataset": "twilio",
                        "session": name,
                        "vad_list": json.dumps(win["vad_list"]),
                    }
                )

    print('VAD Error: ', len(bad_json))
    write_txt(bad_json, 'bad_jsons.txt')

    # Create a DataFrame from all_results
    df_results = pd.DataFrame(all_results)

    # Write the DataFrame back to the csv
    df_results.to_csv(args.csv_path, index=False)
    print("Saved csv file -> ", args.csv_path)
