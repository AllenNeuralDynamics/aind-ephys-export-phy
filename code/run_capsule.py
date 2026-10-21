import warnings

warnings.filterwarnings("ignore")

# GENERAL IMPORTS
import argparse
import json
import logging
import os
import re
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

# Spike times converted per block, to keep memory bounded for billions of spikes
SPIKE_TIME_BLOCK = 100_000_000

# Extensions export_to_phy reads (PCs are not exported)
PHY_EXTENSIONS = (
    "random_spikes",
    "waveforms",
    "templates",
    "template_similarity",
    "spike_amplitudes",
    "quality_metrics",
    "template_metrics",
)


def load_extension_data_in_memory(analyzer) -> None:
    """Read the data of the extensions export_to_phy uses into numpy arrays.

    A lazily loaded analyzer keeps extension data in zarr arrays, which
    export_to_phy indexes like numpy (e.g. amplitudes[:, np.newaxis]) and fails
    on. The spike vector itself stays lazy.
    """
    for name in PHY_EXTENSIONS:
        extension = analyzer.get_extension(name)
        if extension is None:
            continue
        for key, value in extension.data.items():
            # zarr (or other array-like) data; DataFrames and numpy arrays stay
            if not isinstance(value, np.ndarray) and hasattr(value, "shape") and hasattr(value, "dtype"):
                extension.data[key] = np.asarray(value)


SUBSET = "_phy_spikes_subset"
# Selected spikes whose frames are checked against the sorting
SUBSET_CHECK_SPIKES = 10_000


def find_sampled_waveforms(recording_name: str, analyzer) -> Path | None:
    """The waveform sampling capsule's folder for this recording, if it matches.

    The sampled spikes are rows of the sorting's spike vector, so they are only
    used when the folder was made from this sorting: same spike and unit counts,
    unit ids, waveform length, and frames of a sample of the selected spikes.
    """
    folder = next(
        (p for p in data_folder.rglob(f"phy_waveforms_{recording_name}") if p.is_dir()), None
    )
    if folder is None:
        return None
    manifest_file = folder / f"{SUBSET}.manifest.json"
    if not manifest_file.is_file():
        logging.warning(f"\t{folder.name} has no manifest: not using it")
        return None
    manifest = json.loads(manifest_file.read_text())
    source = manifest.get("source_sorting", {})
    spike_vector = analyzer.sorting.to_spike_vector()
    templates_ext = analyzer.get_extension("templates")
    reasons = []
    if manifest.get("format") != "phy_spikes_subset" or manifest.get("format_version") != 1:
        reasons.append("unknown format")
    if source.get("num_spikes") != len(spike_vector):
        reasons.append(f"{source.get('num_spikes')} spikes vs {len(spike_vector)} in the sorting")
    if source.get("unit_ids") != [str(u) for u in analyzer.unit_ids]:
        reasons.append("different unit ids")
    extraction = manifest.get("waveform_extraction", {})
    if templates_ext is not None and (
        extraction.get("nbefore_samples") != templates_ext.nbefore
        or extraction.get("nafter_samples") != templates_ext.nafter
    ):
        reasons.append("waveform length differs from the templates")
    if not reasons:
        rows = np.load(folder / f"{SUBSET}.spikes.npy", mmap_mode="r")
        frames = np.load(folder / f"{SUBSET}.sample_index.npy", mmap_mode="r")
        check = np.unique(np.linspace(0, rows.size - 1, min(rows.size, SUBSET_CHECK_SPIKES)).astype(np.int64))
        if rows.size:
            sample_index = spike_vector["sample_index"]
            check_rows = np.asarray(rows[check])
            # zarr (lazy analyzer) or numpy
            sorting_frames = (
                sample_index.oindex[check_rows] if hasattr(sample_index, "oindex")
                else np.asarray(sample_index)[check_rows]
            )
            if not np.array_equal(sorting_frames, np.asarray(frames[check])):
                reasons.append("selected spikes' frames differ from the sorting")
    if reasons:
        logging.warning(f"\tNot using {folder.name}: {'; '.join(reasons)}")
        return None
    return folder


def convert_spike_times_to_clock(
    phy_folder: Path, timestamps: dict, num_frames: int
) -> str | None:
    """Rewrite spike_times.npy in clock time, from the timestamp alignment capsule.

    Spike times are frames on the pipeline timeline, where acquisition gaps are
    absent. They become samples of clock time since the first frame:
    round((clock(frame) - clock_start) * fs), with clock() the piecewise-linear
    mapping of timestamps_<recording_name>.json. Spike order is unchanged (the
    mapping never decreases), so the other Phy files stay aligned.

    Returns why the conversion was skipped, or None if it was applied.
    """
    if not timestamps.get("has_timestamps", False):
        return "the raw recording has no timestamps"
    if timestamps["pipeline_num_frames"] != num_frames:
        return (
            f"it covers {timestamps['pipeline_num_frames']:,} frames but the "
            f"recording has {num_frames:,}"
        )
    # Raw data in Phy is indexed by frame: shifted spike times would not line up.
    # Without raw data, params.py has dat_path = None or (absolute paths) r'None'
    params_text = (phy_folder / "params.py").read_text()
    if not re.search(r"""^dat_path\s*=\s*(?:None|r?['"]None['"])\s*$""", params_text, re.M):
        return "the export has raw data (dat_path), which is indexed by frame"

    fs = timestamps["sampling_frequency"]
    mapping_frames = np.asarray(timestamps["mapping"]["pipeline_frame"], dtype=np.float64)
    mapping_clock = np.asarray(timestamps["mapping"]["clock_s"], dtype=np.float64)
    clock_start = timestamps["clock_start_s"]

    spike_times = np.load(phy_folder / "spike_times.npy", mmap_mode="r+")
    previous_last = None
    for i0 in range(0, spike_times.shape[0], SPIKE_TIME_BLOCK):
        frames = spike_times[i0 : i0 + SPIKE_TIME_BLOCK, 0]
        clock = np.interp(frames, mapping_frames, mapping_clock)
        converted = np.round((clock - clock_start) * fs).astype(spike_times.dtype)
        if np.any(np.diff(converted) < 0) or (
            previous_last is not None and converted[0] < previous_last
        ):
            raise RuntimeError("Converted spike times are not in order")
        spike_times[i0 : i0 + SPIKE_TIME_BLOCK, 0] = converted
        previous_last = converted[-1]
    spike_times.flush()
    del spike_times
    return None


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
        except Exception as e:
            logging.info(
                f"Spike sorting failed on {recording_name}. Skipping export to phy "
                f"(could not load {postprocessed_folder.name}: {type(e).__name__}: {e})"
            )
            if postprocessed_folder.is_dir():
                contents = sorted(p.name for p in postprocessed_folder.iterdir())[:20]
                logging.info(f"\t{postprocessed_folder.name} contains: {contents}")
            # create an empty result file (needed for pipeline)
            export_output_folder.mkdir()
            error_file = export_output_folder / "error.txt"
            error_file.write_text(
                f"Spike sorting failed.\nCould not load {postprocessed_folder.name}: "
                f"{type(e).__name__}: {e}\n"
            )
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

        load_extension_data_in_memory(analyzer)

        # Waveforms sampled from shared temporal windows (waveform sampling
        # capsule) if present; otherwise the analyzer's random_spikes, as before
        sampled_waveforms = find_sampled_waveforms(recording_name, analyzer)
        if sampled_waveforms is not None:
            logging.info(f"\tUsing sampled waveforms from {sampled_waveforms.name}")
        else:
            logging.info("\tUsing the analyzer's random_spikes waveforms")

        sexp.export_to_phy(
            analyzer,
            output_folder=export_output_folder,
            compute_pc_features=False,
            add_waveforms=sampled_waveforms is None,
            copy_binary=SAVE_BINARY,
            additional_properties=additional_properties,
            dtype=dtype,
            verbose=False,
        )
        if sampled_waveforms is not None:
            for subset_file in sorted(sampled_waveforms.glob(f"{SUBSET}.*")):
                shutil.copy(subset_file, export_output_folder / subset_file.name)
            export_notes += f"\n- Waveforms from {sampled_waveforms.name}.\n"

        # Spike times in clock time, if the timestamp alignment capsule ran;
        # otherwise they stay in frames, as before
        timestamps_file = next(
            data_folder.rglob(f"timestamps_{recording_name}.json"), None
        )
        if timestamps_file is None:
            logging.info("\tNo timestamps file: spike_times stay in frames")
        else:
            timestamps = json.loads(timestamps_file.read_text())
            skipped = convert_spike_times_to_clock(
                export_output_folder, timestamps, analyzer.get_num_samples()
            )
            # Kept with the Phy output either way (frame to clock-time mapping, gaps)
            shutil.copy(timestamps_file, export_output_folder / "timestamps.json")
            if skipped is None:
                note = (
                    f"spike_times converted to clock time from {timestamps_file.name} "
                    f"({len(timestamps['gaps'])} gap(s), "
                    f"{timestamps['total_gap_s'] / 3600:.2f} h)"
                )
                logging.info(f"\t{note}")
                export_notes += f"\n- {note}.\n"
            else:
                logging.warning(
                    f"\tNot converting spike_times with {timestamps_file.name}: {skipped}. "
                    "timestamps.json is in the Phy folder, but spike_times stay in frames"
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
