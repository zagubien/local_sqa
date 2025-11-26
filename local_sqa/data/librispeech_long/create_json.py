import os
from pathlib import Path

import click
from paderbox.io import dump_json
from paderbox.io.audioread import audio_length

DATABASE_ROOT = Path(os.environ.get("DATABASE_ROOT", ""))

splits = { 
    "dev_clean": "dev-clean",
    "dev_other": "dev-other",
    "test_clean": "test-clean",
    "test_other": "test-other",
}

def collect_examples(database_path: Path, splits: dict):

    datasets= { }

    for json_key, folder_name in splits.items(): 
        split_dir = database_path / folder_name

        if not split_dir.exists():
            raise RuntimeError(f"Split directory not found: {split_dir}")
        
        split_examples = {}

        for speaker_dir in split_dir.iterdir():
            if not speaker_dir.is_dir():
                continue

            for chapter_dir in speaker_dir.iterdir():
                if not chapter_dir.is_dir():
                    continue

                for flac_file in chapter_dir.glob("*.flac"):
                    example_id = flac_file.stem  

                    txt_file = flac_file.with_suffix(".txt") 

                    if not txt_file.exists():
                        continue

                    transcription = txt_file.read_text().strip()
                    num_samples = audio_length(str(flac_file), unit=  "samples")

                    example = {
                        "example_id" : example_id,
                        "audio_path": {"observation": str(flac_file)},
                        "num_samples": num_samples,
                        "sampling_rate": 16000,
                        "transcription": transcription,
                    }

                    split_examples[example_id] = example

        datasets[json_key] = split_examples

    return datasets
        

def create_json(database_path: Path):
    datasets = collect_examples(database_path, splits)
    database ={"datasets": datasets}
    return database 

def save_json(database, json_path: Path):
    dump_json(
        database,
        json_path,
        create_path=True,
        indent=4,
        ensure_ascii=False,
    )


@click.command()
@click.option(
    '--json-path', '-j',
    default='librispeech_long.json',
    help=(
        'Output path for the generated JSON file. If the '
        'file exists, it gets overwritten. Defaults to '
        '"librispeech_long.json".'
    ),
    type=click.Path(dir_okay=False, writable=True),
)

@click.option(
    '--database-path', '-db',
    default=DATABASE_ROOT / 'librispeech_long',
    help='Path where the database is located.',
    type=click.Path(),
)


def main(json_path, database_path):
    database_path = Path(database_path).absolute()
    json_path= Path(json_path)
    database = create_json(database_path)
    save_json(database, json_path)

    print(f"erfolg")

if __name__ == "__main__":
    main()