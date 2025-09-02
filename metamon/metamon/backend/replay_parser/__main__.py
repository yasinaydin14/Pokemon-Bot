import os
import glob
import random
import tqdm

import metamon
from metamon.backend.replay_parser.parse_replays import ReplayParser
from metamon.backend.team_prediction.predictor import ALL_PREDICTORS

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--format",
        type=str,
        choices=metamon.SUPPORTED_BATTLE_FORMATS,
        required=True,
    )
    parser.add_argument(
        "--raw_replay_dir",
        default=None,
        help="Path to raw replay dataset folder. Defaults to the latest huggingface version.",
    )
    parser.add_argument("--max", type=int, help="Parse up to this many replays.")
    parser.add_argument(
        "--filter_by_code",
        help="Skip to a specific game id. For example: `gen4ubers-1101300080`",
    )
    parser.add_argument(
        "--start_from",
        type=int,
        default=0,
        help="Start parsing from this index of the dataset (skip replays you've already checked)",
    )
    parser.add_argument(
        "--end_after",
        type=int,
        default=None,
        help="Stop parsing after this many replays",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Prints the raw replay stream during parsing (useful for debugging)",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Number of parallel parser processes to run",
    )
    parser.add_argument(
        "--replay_stats_dir",
        default=None,
        help="Directory for existing replay team statistics. If not provided, will default to the official version in the cached parsed replay dataset from hf.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for output .npz files. `None` runs w/o saving to disk. Data will be saved to {--output_dir}/gen{gen}{format}",
    )
    parser.add_argument(
        "--team_predictor",
        type=str,
        choices=list(ALL_PREDICTORS.keys()),
        default="NaiveUsagePredictor",
        help="Team predictor to use",
    )
    parser.add_argument(
        "--team_output_dir",
        default=None,
        help="Directory for output .team files. `None` runs w/o saving to disk. Data will be saved to {--team_output_dir}/gen{gen}{format}_teams",
    )
    args = parser.parse_args()

    if args.raw_replay_dir is None:
        args.raw_replay_dir = os.path.join(metamon.METAMON_CACHE_DIR, "raw-replays")

    gen = args.format[:4]
    format = args.format[4:].lower()
    path = os.path.join(args.raw_replay_dir, gen, format)
    filenames = glob.glob(f"{path}/**/*.json", recursive=True)
    random.shuffle(filenames)
    if args.filter_by_code is not None:
        filenames = [f for f in filenames if args.filter_by_code in f]
    if args.start_from is not None:
        filenames = filenames[args.start_from :]
    if args.end_after is not None:
        filenames = filenames[: args.end_after]
    if args.max is not None:
        filenames = filenames[: args.max]
    output_dir = os.path.join(args.output_dir, args.format) if args.output_dir else None
    team_output_dir = (
        os.path.join(args.team_output_dir, args.format)
        if args.team_output_dir
        else None
    )
    parser = ReplayParser(
        replay_output_dir=output_dir,
        team_output_dir=team_output_dir,
        verbose=args.verbose,
        team_predictor=ALL_PREDICTORS[args.team_predictor](
            replay_stats_dir=args.replay_stats_dir
        ),
    )
    if args.processes > 1:
        random.shuffle(filenames)
        parser.parse_parallel(filenames, args.processes)
    else:
        for filename in tqdm.tqdm(filenames):
            parser.parse_replay(filename)
        errors = parser.summarize_errors()
        for fb, sub in errors.items():
            print(f"{fb} Errors:")
            for i, (err, c) in enumerate(sub.items()):
                print(f"\t{i + 1}. {err}: {c}")
