import os
import re
import json
import datetime
import functools
import warnings
from typing import Optional

from termcolor import colored
import metamon
from metamon.backend.team_prediction.usage_stats.format_rules import (
    get_valid_pokemon,
    Tier,
)
from metamon.backend.replay_parser.str_parsing import pokemon_name
from metamon.backend.showdown_dex.dex import Dex


TIER_MAP = {
    "ubers": Tier.UBERS,
    "ou": Tier.OU,
    "uu": Tier.UU,
    "ru": Tier.RU,
    "nu": Tier.NU,
    "pu": Tier.PU,
    "lc": Tier.LC,
}

EARLIEST_USAGE_STATS_DATE = datetime.date(2014, 1, 1)
LATEST_USAGE_STATS_DATE = datetime.date(2025, 7, 1)


def parse_pokemon_moveset(file_path):
    moveset_data_list = {
        "name": [],
        "count": [],
        "abilities": [],
        "items": [],
        "spreads": [],
        "moves": [],
        "tera_types": [],
        "teammates": [],
        "checks": [],
    }

    def p_name(data_cache):
        name = data_cache[0][2:-2].strip()
        moveset_data_list["name"].append(name)
        return moveset_data_list

    def p_count(data_cache):
        count = int(data_cache[0][2:-2].split(":")[1].strip())
        moveset_data_list["count"].append(count)
        return moveset_data_list

    def p_abilities(data_cache):
        _abilities = {}
        assert "Abilities" in data_cache[0], "Abilities not found"
        for line in data_cache[1:]:
            line_split = line[2:-2].strip().split()
            name = " ".join(line_split[:-1])
            percent = line_split[-1]
            # percent is in string format of xx%
            _abilities[name] = float(percent[:-1]) / 100
        moveset_data_list["abilities"].append(_abilities)
        # assert _abilities, "abilities is empty"
        return moveset_data_list

    def p_items(data_cache):
        _items = {}
        assert "Items" in data_cache[0], "Items not found"
        for line in data_cache[1:]:
            line_split = line[2:-2].strip().split()
            name = " ".join(line_split[:-1])
            percent = line_split[-1]
            # percent is in string format of xx%
            _items[name] = float(percent[:-1]) / 100
        moveset_data_list["items"].append(_items)
        # assert _items, "items is empty"
        return moveset_data_list

    def p_spreads(data_cache):
        _spreads = {}
        assert "Spreads" in data_cache[0], "Spreads not found"
        for line in data_cache[1:]:
            nature_ev, percent = line[2:-2].strip().split()
            # percent is in string format of xx%
            _spreads[nature_ev] = float(percent[:-1]) / 100
        moveset_data_list["spreads"].append(_spreads)
        # assert _spreads, "spreads is empty"
        return moveset_data_list

    def p_moves(data_cache):
        _moves = {}
        assert "Moves" in data_cache[0], "Moves not found"
        for line in data_cache[1:]:
            line_split = line[2:-2].strip().split()
            name = " ".join(line_split[:-1])
            percent = line_split[-1]
            _moves[name] = float(percent[:-1]) / 100
        # assert _moves, "moves is empty"
        moveset_data_list["moves"].append(_moves)
        return moveset_data_list

    def p_tera_types(data_cache):
        _tera_types = {}
        assert "Tera Types" in data_cache[0], "Tera Types not found"
        for line in data_cache[1:]:
            line_split = line[2:-2].strip().split()
            name = " ".join(line_split[:-1])
            percent = line_split[-1]
            _tera_types[name] = float(percent[:-1]) / 100
        moveset_data_list["tera_types"].append(_tera_types)
        return moveset_data_list

    def p_teammates(data_cache):
        _teammates = {}
        assert "Teammates" in data_cache[0], "Teammates not found"
        for line in data_cache[1:]:
            line_split = line[2:-2].strip().split()
            name = " ".join(line_split[:-1])
            percent = line_split[-1]
            if percent.startswith("+") or percent.startswith("-"):
                percent = percent[1:]
            # sometimes the data will have a duplicate line and cause error, just skip that line
            try:
                _teammates[name] = float(percent[:-1]) / 100
            except:
                continue
        # assert _teammates, "teammates is empty"
        moveset_data_list["teammates"].append(_teammates)
        return moveset_data_list

    def p_checks(data_cache):
        _checks = {}
        assert "Checks and Counters" in data_cache[0], "Checks and Counters not found"
        for i in range(1, len(data_cache), 2):
            line = data_cache[i]
            line_split = line[2:-2].strip().split()
            name = " ".join(line_split[:-2])
            percent = line_split[-2]
            # percent is in string format of xx%
            _checks[name] = float(percent) / 100
        moveset_data_list["checks"].append(_checks)
        # assert _checks, "checks is empty"
        return moveset_data_list

    with open(file_path, "r") as file:
        file_content = file.read()

    if "Tera Types" in file_content:
        section_order = [
            p_name,
            p_count,
            p_abilities,
            p_items,
            p_spreads,
            p_moves,
            p_tera_types,
            p_teammates,
            p_checks,
            lambda _: None,
        ]
    else:
        section_order = [
            p_name,
            p_count,
            p_abilities,
            p_items,
            p_spreads,
            p_moves,
            p_teammates,
            p_checks,
            lambda _: None,
        ]

    lines = file_content.split("\n")
    current_section = -1

    data_cache = []
    for line in lines:
        # print(line)
        if line.startswith(" +-"):
            section_order[current_section](data_cache)
            current_section = (current_section + 1) % len(section_order)
            data_cache = []
        elif line.startswith(" |"):
            data_cache.append(line.strip())

    n = len(moveset_data_list["name"])
    if len(moveset_data_list["tera_types"]) == 0:
        moveset_data_list["tera_types"] = [{"Nothing": 1.0} for _ in range(n)]

    moveset_data = {}
    for i in range(n):
        _name = moveset_data_list["name"][i]
        _count = moveset_data_list["count"][i]
        _abilities = moveset_data_list["abilities"][i]
        _items = moveset_data_list["items"][i]
        _spreads = moveset_data_list["spreads"][i]
        _moves = moveset_data_list["moves"][i]
        _tera_types = moveset_data_list["tera_types"][i]
        _teammates = moveset_data_list["teammates"][i]
        _checks = moveset_data_list["checks"][i]
        moveset_data[_name] = {
            "count": _count,
            "abilities": _abilities,
            "items": _items,
            "spreads": _spreads,
            "moves": _moves,
            "tera_types": _tera_types,
            "teammates": _teammates,
            "checks": _checks,
        }
    return moveset_data


def merge_movesets(movesets):
    result = {}
    for moveset in movesets:
        for pokemon, data in moveset.items():
            # add up count first
            count = data["count"]
            if pokemon not in result:
                result[pokemon] = {}
                result[pokemon]["count"] = 0
            for key, value in data.items():
                if key == "count":
                    result[pokemon][key] += value
                    continue
                # key is the entry name, e.g. "abilities", "items", "spreads", "moves", "teammates", "checks"
                if key not in result[pokemon]:
                    result[pokemon][key] = {}
                # k is the counted thing, v is the percentage
                for k, v in value.items():
                    if k not in result[pokemon][key]:
                        result[pokemon][key][k] = 0
                    result[pokemon][key][k] += v * count

    # recalculating the percentage
    for pokemon, data in result.items():
        count = data["count"]
        for key, value in data.items():
            if key != "count":
                for k, v in value.items():
                    result[pokemon][key][k] /= count

    return result


class SmogonStat:
    def __init__(
        self,
        format: str,
        raw_stats_dir: str,
        date=None,
        rank=None,
        verbose: bool = True,
    ) -> None:
        if date and type(date) == str:
            dates = [date]
        elif date and type(date) == list:
            dates = date
        else:
            dates = os.listdir(raw_stats_dir)

        self.data_paths = [os.path.join(raw_stats_dir, date) for date in dates]
        self.format = format
        self.rank = rank
        self.verbose = verbose

        self._movesets = {}
        self._inclusive = {}
        self._usage = None
        self._load()
        self._name_conversion = {
            pokemon_name(pokemon): pokemon for pokemon in self._movesets.keys()
        }

    def _load(self):
        moveset_paths = []
        for data_path in self.data_paths:
            moveset_path = os.path.join(data_path, "moveset")
            if os.path.exists(moveset_path):
                moveset_paths.append(moveset_path)

        if len(moveset_paths) == 0:
            print(f"No moveset data found for {self.format} in {self.data_paths}")
            self._movesets = {}
            return

        _movesets = []
        for moveset_path in moveset_paths:
            format_data = [
                x for x in os.listdir(moveset_path) if x.startswith(self.format + "-")
            ]
            if self.rank is not None:
                format_data = [x for x in format_data if self.rank in x]
            _movesets += [
                parse_pokemon_moveset(os.path.join(moveset_path, x))
                for x in format_data
            ]
        self._movesets = {
            pokemon_name(k): v for k, v in merge_movesets(_movesets).items()
        }

    def pretty_print(self, key):
        data = self[key]
        print(f"------ {key} {data['count']} ------\n")
        print("\tMoveset:")
        sorted_moves = sorted(data["moves"].items(), key=lambda x: x[1], reverse=True)
        for i, (move, usage) in enumerate(sorted_moves[:10]):
            print(f"\t\t{i+1}. {move} ({usage * 100: .1f}%)")
        print("\n\tTeammates:")
        sorted_mates = sorted(
            data["teammates"].items(), key=lambda x: x[1], reverse=True
        )
        for i, (mate, usage) in enumerate(sorted_mates[:5]):
            print(f"\t\t{i+1}. {mate} ({usage * 100: .1f}%)")
        print("\n\tChecks:")
        sorted_checks = sorted(data["checks"].items(), key=lambda x: x[1], reverse=True)
        for i, (check, usage) in enumerate(sorted_checks[:5]):
            print(f"\t\t{i+1}. {check} ({usage * 100: .1f}%)")

    def remove_banned_pm(self):
        valid_pm_dict = get_valid_pokemon(self.format)
        tier = TIER_MAP[self.format[4:]]
        valid_pm = []
        # get all pokemon valid in this tier
        for t in valid_pm_dict.keys():
            if t >= tier:
                valid_pm.extend(valid_pm_dict[t])

        if self.verbose:
            print(f"Total {len(valid_pm)} valid pokemon for {self.format}")
        # remove pokemon that not in this tier
        for pm in list(self._movesets.keys()):
            if re.sub(r"[^a-zA-Z0-9]", "", pm) not in valid_pm:
                if self.verbose:
                    print(f"Remove {pm} from {self.format}")
                del self._movesets[pm]

    @property
    def movesets(self):
        return self._movesets

    @property
    def usage(self):
        if self._usage is None:
            # create a list of pokemon names, sorted by count
            self._usage = list(
                sorted(
                    self._movesets.keys(),
                    key=lambda x: self._movesets[x]["count"],
                    reverse=True,
                )
            )
        return self._usage


def load_between_dates(
    dir_path: str, start_year: int, start_month: int, end_year: int, end_month: int
) -> dict:
    start_date = datetime.date(start_year, start_month, 1)
    end_date = datetime.date(end_year, end_month, 1)
    selected_data = []
    for json_file in os.listdir(dir_path):
        year, month = json_file.replace(".json", "").split("-")
        date = datetime.date(year=int(year), month=int(month), day=1)
        if not start_date <= date <= end_date:
            continue
        with open(os.path.join(dir_path, json_file), "r") as file:
            data = json.load(file)
        selected_data.append(data)
    if not selected_data:
        warnings.warn(
            colored(
                f"No Showdown usage stats found in {dir_path} between {start_date} and {end_date}",
                "red",
            )
        )
    return merge_movesets(selected_data)


class PreloadedSmogonUsageStats(SmogonStat):
    def __init__(
        self,
        format,
        start_date: datetime.date,
        end_date: datetime.date,
        verbose: bool = True,
    ):
        self.format = format.strip().lower()
        self.rank = None
        self.start_date = start_date
        self.end_date = end_date
        self.verbose = verbose
        self._usage = None
        gen, tier = int(self.format[3]), self.format[4:]
        self.gen = gen
        usage_stats_path = metamon.data.download.download_usage_stats(gen)
        movesets_path = os.path.join(
            usage_stats_path, "movesets_data", f"gen{gen}", f"{tier}"
        )
        inclusive_path = os.path.join(
            usage_stats_path, "movesets_data", f"gen{gen}", "all_tiers"
        )
        # data is split by year and month
        if not os.path.exists(movesets_path) or not os.path.exists(inclusive_path):
            raise FileNotFoundError(
                f"Movesets data not found for {format}. Run `python -m metamon download usage-stats` to download the data."
            )
        self._movesets = load_between_dates(
            movesets_path,
            start_year=start_date.year,
            start_month=start_date.month,
            end_year=end_date.year,
            end_month=end_date.month,
        )
        self._inclusive = load_between_dates(
            inclusive_path,
            start_year=EARLIEST_USAGE_STATS_DATE.year,
            start_month=EARLIEST_USAGE_STATS_DATE.month,
            end_year=LATEST_USAGE_STATS_DATE.year,
            end_month=LATEST_USAGE_STATS_DATE.month,
        )

    def _load(self):
        pass

    def _inclusive_search(self, key):
        # check the stats for this specific tier and time period first
        key_id = pokemon_name(key)
        recent = self._movesets.get(key_id, {})
        alltime = self._inclusive.get(key_id, {})
        if not (recent or alltime):
            return None

        if recent and alltime:
            # use the alltime stats to selectively get keys that exist
            # in recent but are unhelpful for team prediction.
            no_info = {"Nothing": 1.0}
            for key, value in recent.items():
                if value == no_info:
                    if alltime.get(key, {}) != no_info:
                        recent[key] = alltime[key]
        return recent if recent else alltime

    def __getitem__(self, key):
        entry = Dex.from_gen(self.gen).get_pokedex_entry(key)
        species, base_species = entry.get("name", key), entry.get("baseSpecies", key)
        lookup = self._inclusive_search(species)
        if lookup is not None:
            return lookup
        lookup = self._inclusive_search(base_species)
        if lookup is not None:
            return lookup
        raise KeyError(f"Pokemon {key} not found in {self.format}")


def get_usage_stats(
    format,
    start_date: Optional[datetime.date] = None,
    end_date: Optional[datetime.date] = None,
) -> PreloadedSmogonUsageStats:
    if start_date is None or start_date < EARLIEST_USAGE_STATS_DATE:
        start_date = EARLIEST_USAGE_STATS_DATE
    else:
        # force to start of months to prevent cache miss (we only have monthly stats anyway)
        start_date = datetime.date(start_date.year, start_date.month, 1)
    if end_date is None or end_date > LATEST_USAGE_STATS_DATE:
        end_date = LATEST_USAGE_STATS_DATE
    else:
        # force to start of months to prevent cache miss (we only have monthly stats anyway)
        end_date = datetime.date(end_date.year, end_date.month, 1)
    return _cached_smogon_stats(format, start_date, end_date)


@functools.lru_cache(maxsize=64)
def _cached_smogon_stats(format, start_date: datetime.date, end_date: datetime.date):
    print(f"Loading usage stats for {format} between {start_date} and {end_date}")
    return PreloadedSmogonUsageStats(
        format=format, start_date=start_date, end_date=end_date, verbose=False
    )


if __name__ == "__main__":
    stats = get_usage_stats(
        "gen9ou", datetime.date(2023, 1, 1), datetime.date(2025, 6, 1)
    )
    print(len(stats.usage))
    for mon in sorted(
        stats.movesets.keys(), key=lambda m: stats[m]["count"], reverse=True
    )[:5]:
        stats.pretty_print(mon)
        print()
