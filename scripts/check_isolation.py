"""
Задание 7 (вспомогательный инструмент): проверка, какие сущности базы знаний
действительно упоминаются ТОЛЬКО в собственном документе - то есть пригодны
для честного gap-теста в evaluate.py (GAP_SOURCES).

Находка код-ревью: первая версия такой проверки искала точное совпадение
полного имени ("Барон Илиан Кросс") и пропускала склоненные формы
("Барона Илиана Кросса", "Илиану Кроссу" и т.д.), из-за чего два из трех
выбранных "изолированных" объектов на самом деле упоминались в других
документах. Здесь используется поиск по основе имени/фамилии (без учета
падежных окончаний) и явно исключается собственный документ сущности.

Запуск:
    python scripts/check_isolation.py
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEARCH_DIRS = [ROOT / "knowledge_base", ROOT / "docs"]

# Основа имени/фамилии для каждой сущности (без падежных окончаний).
# Ключ - "человекочитаемое" имя, значение - regex-паттерн основы.
ENTITY_STEMS = {
    "Каэл Виндраннер": r"Виндраннер[а-я]*",
    "Ксарн Велгор": r"Велгор[а-я]*",
    "Сера Номира": r"Номир[а-я]*",
    "Даск Оррин": r"Оррин[а-я]*",
    "Торен Халвекс": r"Халвекс[а-я]*",
    "Виндрал": r"Виндрал(?!\w)[а-я]*",
    "Мордрек Тейн": r"Тейн[а-я]*",
    "Врошан": r"Врошан[а-я]*",
    "Коррат Венн": r"Венн[а-я]*",
    "Барон Илиан Кросс": r"Илиан[а-я]*\s+Кросс[а-я]*",
    "Карсис Прайм": r"Карсис\s+Прайм[а-я]*",
    "Солмара": r"Солмар[а-я]*",
    "Вэлторн": r"Вэлторн[а-я]*",
    "Мурквейл": r"Мурквейл[а-я]*",
    "Нексарион": r"Нексарион[а-я]*",
    "Сильван": r"Сильван[а-я]*(?!ский)",
    "Войд-Ядро": r"Войд-Ядр[а-я]*",
    "Синт-Поток": r"Синт.Поток[а-я]*",
    "Флюкс-клинок": r"Флюкс.клинк?[а-я]*",
}

# Файл, в котором сущность считается "своей" (не в счет при подсчете
# посторонних упоминаний) - находится по slug имени файла.
OWN_FILE_HINTS = {
    "Каэл Виндраннер": "kael_vindranner",
    "Ксарн Велгор": "ksarn_velgor",
    "Сера Номира": "sera_nomira",
    "Даск Оррин": "dask_orrin",
    "Торен Халвекс": "toren_halveks",
    "Виндрал": "vindral",
    "Мордрек Тейн": "mordrek_tein",
    "Врошан": "vroshan",
    "Коррат Венн": "korrat_venn",
    "Барон Илиан Кросс": "ilian_kross",
    "Карсис Прайм": "karsis_prime",
    "Солмара": "solmara",
    "Вэлторн": "veltorn",
    "Мурквейл": "murkveil",
    "Нексарион": "neksarion",
    "Сильван": "silvan",
    "Войд-Ядро": "voyd_yadro",
    "Синт-Поток": "synt_potok",
    "Флюкс-клинок": "flyuks_klinok",
}


def all_files() -> list[Path]:
    files = []
    for d in SEARCH_DIRS:
        files.extend(sorted(d.glob("*.md")))
    return files


def main() -> None:
    files = all_files()
    texts = {p: p.read_text(encoding="utf-8") for p in files}

    print(f"{'Сущность':30s} {'посторонних упоминаний':24s} файлы")
    isolated = []
    for name, pattern in ENTITY_STEMS.items():
        own_hint = OWN_FILE_HINTS[name]
        rx = re.compile(pattern)
        outside = []
        for p, t in texts.items():
            if own_hint in p.stem:
                continue
            if rx.search(t):
                outside.append(p.name)
        status = "ИЗОЛИРОВАНА" if not outside else f"{len(outside)} упоминаний"
        print(f"{name:30s} {status:24s} {outside}")
        if not outside:
            isolated.append(name)

    print("\nПригодны для GAP_SOURCES (0 посторонних упоминаний):")
    for name in isolated:
        print(" -", name, "->", OWN_FILE_HINTS[name])


if __name__ == "__main__":
    main()
