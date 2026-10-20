import warnings

warnings.filterwarnings("ignore")

# GENERAL IMPORTS
import argparse
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# SPIKEINTERFACE
import spikeinterface as si
import spikeinterface.exporters as sexp
from aind_data_schema.components.identifiers import Code

# AIND
from aind_data_schema.core.processing import DataProcess, ProcessStage
from aind_data_schema_models.process_names import ProcessName

URL = "https://github.com/AllenNeuralDynamics/aind-ephys-export-phy"
VERSION = "2.0"

data_folder = Path("../data/")
scratch_folder = Path("../scratch")
results_folder = Path("../results/")

# Define argument parser
parser = argparse.ArgumentParser(description="Curate ecephys data")

save_binary_group = parser.add_mutually_exclusive_group()
save_binary_help = "Whether to save the recording to binary"
save_binary_group.add_argument(
    "--save-binary", action="store_true", help=save_binary_help
)
save_binary_group.add_argument("static_save_binary", nargs="?", help=save_binary_help)

n_jobs_group = parser.add_mutually_exclusive_group()
n_jobs_help = (
    "Number of jobs to use for parallel processing. Default is -1 (all available cores). "
    "It can also be a float between 0 and 1 to use a fraction of available cores"
)
n_jobs_group.add_argument("static_n_jobs", nargs="?", default="-1", help=n_jobs_help)
n_jobs_group.add_argument("--n-jobs", default="-1", help=n_jobs_help)

parser.add_argument(
    "--params",
    default=None,
    help="Path to the parameters file or JSON string. If given, it will override all other arguments.",
)


if __name__ == "__main__":
    ####### Export to phy ########
    export_notes = ""
    t_export_start_all = time.perf_counter()

    args = parser.parse_args()

    SAVE_BINARY = (
        args.static_save_binary.lower() == "true"
        if args.static_save_binary
        else args.save_binary
    )
    N_JOBS = args.static_n_jobs or args.n_jobs
    N_JOBS = int(N_JOBS) if not N_JOBS.startswith("0.") else float(N_JOBS)
    PARAMS = args.params

    # Use CO_CPUS/N_JOBS_EXT env variable if available
    N_JOBS_EXT = os.getenv("CO_CPUS") or os.getenv("N_JOBS_EXT")
    N_JOBS = int(N_JOBS_EXT) if N_JOBS_EXT is not None else N_JOBS

    data_process_prefix = "data_process_export_phy"

    # spawn: s3fs (recordings stored on S3) is not fork-safe
    job_kwargs = {"n_jobs": N_JOBS, "progress_bar": False, "mp_context": "spawn"}
    si.set_global_job_kwargs(**job_kwargs)

    ecephys_sorted_folders = [
        p
        for p in data_folder.iterdir()
        if p.is_dir()
        and "ecephys" in p.name
        or "behavior" in p.name
        and "sorted" in p.name
    ]

    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")
    logging.info("Running export to Phy")
    logging.info(f"\tSAVE_BINARY: {SAVE_BINARY}")

    if len(ecephys_sorted_folders) > 0:
        # capsule mode
        assert len(ecephys_sorted_folders) == 1, "Attach one sorted asset at a time"
        ecephys_sorted_folder = ecephys_sorted_folders[0]
        postprocessed_base_folder = ecephys_sorted_folder / "postprocessed"
    elif (data_folder / "postprocessing_pipeline_output_test").is_dir():
        logging.info(
            "\n*******************\n**** TEST MODE ****\n*******************\n"
        )
        preprocessed_base_folder = data_folder / "preprocessing_pipeline_output_test"
        postprocessed_base_folder = data_folder / "postprocessing_pipeline_output_test"
        curated_base_folder = data_folder / "curation_pipeline_output_test"
    else:
        preprocessed_base_folder = data_folder
        postprocessed_base_folder = data_folder
        curated_base_folder = data_folder

    postprocessed_folders = [
        p for p in postprocessed_base_folder.iterdir() if "postprocessed_" in p.name
    ]
    for postprocessed_folder in postprocessed_folders:
        datetime_start_export = datetime.now()
        t_export_start = time.perf_counter()
        # Remove 'postprocessed_' for recording_name
        recording_name = postprocessed_folder.name[len("postprocessed_") :]
        # Remove ".zarr"
        recording_name = recording_name[: recording_name.find(".zarr")]
        export_output_process_json = (
            results_folder / f"{data_process_prefix}_{recording_name}.json"
        )
        export_output_folder = results_folder / f"phy_{recording_name}"

        try:
            # lazy: spikes stay on disk until exported; a non-lazy load holds 2
            # copies of the spike vector, which does not fit for long recordings
            analyzer = si.load_sorting_analyzer(postprocessed_folder, lazy=True)
            logging.info(f"Exporting recording: {recording_name}")
        except Exception:
            logging.info(
                f"Spike sorting failed on {recording_name}. Skipping export to phy"
            )
            # create an empty result file (needed for pipeline)
            export_output_folder.mkdir()
            error_file = export_output_folder / "error.txt"
            error_file.write_text("Spike sorting failed.")
            continue

        # Load preprocessed data
        binary_json_file = preprocessed_base_folder / f"binary_{recording_name}.json"
        preprocessed_json_file = (
            preprocessed_base_folder / f"preprocessed_{recording_name}.json"
        )
        motion_corrected_folder = preprocessed_base_folder / f"motion_{recording_name}"

        recording = None
        # Without copying the binary, Phy only gets raw data from a local binary
        # recording, never from one stored on S3: skip loading it from S3 then
        skip_recording = (
            not SAVE_BINARY
            and binary_json_file.is_file()
            and "s3://" in binary_json_file.read_text()
        )
        try:
            if skip_recording:
                logging.info("\tRecording is on S3 and not saved to binary: not loaded")
            elif binary_json_file.is_file():
                recording = si.load(
                    binary_json_file, base_folder=preprocessed_base_folder
                )
                logging.info("\tLoaded binary recording from JSON")
            else:
                recording = si.load(
                    preprocessed_base_folder / f"preprocessed_{recording_name}"
                )
            if recording is None and not skip_recording:
                try:
                    if preprocessed_json_file.is_file():
                        logging.info("\tLoading lazy recording from JSON")
                        recording = si.load(
                            preprocessed_json_file, base_folder=data_folder
                        )
                except:
                    logging.info("Could not load lazy preprocessed recording")
        except Exception as e:
            logging.info(f"Could not load recording: {e}")

        dtype = None
        if recording is not None:
            analyzer._recording = recording
        else:
            dtype = "int16"

        # load additional properties from curated folder
        label_file = curated_base_folder / f"unit_labels_{recording_name}.csv"
        additional_properties = []
        if label_file.is_file():
            # load labels
            labels_df = pd.read_csv(label_file)
            for label in labels_df.columns:
                additional_properties.append(label)
                analyzer.set_sorting_property(label, labels_df[label], save=False)
        if len(additional_properties) > 0:
            logging.info(
                f"\tAdding {len(additional_properties)} additional properties from curation labels"
            )

        if SAVE_BINARY and not (
            analyzer.has_recording() or analyzer.has_temporary_recording()
        ):
            logging.info("Cannot save binary, since recording is not available.")
            SAVE_BINARY = False

        sexp.export_to_phy(
            analyzer,
            output_folder=export_output_folder,
            compute_pc_features=False,
            add_waveforms=True,
            copy_binary=SAVE_BINARY,
            additional_properties=additional_properties,
            dtype=dtype,
            verbose=False,
        )

        # Copy curation.json to output folder
        curation_file = curated_base_folder / f"curation_{recording_name}.json"
        if curation_file.is_file():
            logging.info("\tSaving curation.json to output folder")
            shutil.copy(curation_file, export_output_folder / "curation.json")

        t_export_end = time.perf_counter()
        elapsed_time_export = np.round(t_export_end - t_export_start, 2)

        export_process = DataProcess(
            process_type=ProcessName.EPHYS_CURATION,
            stage=ProcessStage.PROCESSING,
            name="Ephys curation",
            experimenters=["AIND Pipeline"],
            code=Code(
                url=URL,
                version=VERSION,  # either release or git commit
                parameters={},
            ),
            start_date_time=datetime_start_export,
            end_date_time=datetime_start_export
            + timedelta(seconds=np.floor(elapsed_time_export)),
            output_path=str(results_folder),
            output_parameters={},
            notes=export_notes,
        )
        with open(export_output_process_json, "w") as f:
            f.write(export_process.model_dump_json(indent=3))

    t_export_end_all = time.perf_counter()
    elapsed_time_export_all = np.round(t_export_end_all - t_export_start_all, 2)
    logging.info(f"EXPORT TO PHY time: {elapsed_time_export_all}s")
