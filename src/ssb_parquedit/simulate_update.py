"""Simulerer 5 samtidige brukere som oppdaterer samme tabell via parquedit.

Krever tilgang til parquedit og PostgreSQL (DuckLake-backend).
Kjør i DaplaLab med riktige miljøvariabler satt.

Bruk:
    python simulate_concurrent_edits.py

Forventet output FØR retry er implementert:
    Noen brukere feiler med "TransactionContext Error: Conflict on update!"

Forventet output ETTER retry er implementert:
    Alle 5 brukere får OK, med retry-logglinjer i mellom.
"""

import logging
import threading
import time
from unittest.mock import patch

import pandas as pd
import ssb_parquedit.dml as dml_module
from ssb_parquedit import ParquEdit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TABLE_NAME = "concurrent_edit_test"
USERS = ["paven", "trump", "putin", "jonas", "haaland"]

# Thread-local storage for simulert bruker — patch settes én gang globalt,
# og _simulated_get_dapla_user() returnerer riktig bruker per tråd.
_thread_local = threading.local()


def _simulated_get_dapla_user() -> str:
    """Returnerer simulert bruker for gjeldende tråd."""
    return getattr(_thread_local, "dapla_user", "")


def setup_table(pe: ParquEdit) -> int:
    """Opprett testtabell og sett inn én rad. Returnerer rowid."""
    if not pe.exists(TABLE_NAME):
        df = pd.DataFrame([{"name": "Test Person", "salary": 50000}])
        pe.create_table(TABLE_NAME, source=df, product_name="concurrent_test", fill=True)

    result = pe.view(TABLE_NAME, columns=["rowid", "name", "salary"])
    rowid = int(result["rowid"].iloc[0])
    logger.info("Tabell opprettet — rowid=%d, startlønn=50000", rowid)
    return rowid


def simulate_user(
    rowid: int,
    user: str,
    new_salary: int,
    results: list,
    barrier: threading.Barrier,
) -> None:
    """Én bruker-tråd: vent til alle er klare, deretter kall edit().

    Hver tråd oppretter sin egen ParquEdit-instans, slik som i produksjon
    der hver notebook-session har sin egen instans.
    """
    _thread_local.dapla_user = user
    pe = ParquEdit()

    barrier.wait()  # alle 5 starter samtidig

    t_start = time.perf_counter()
    try:
        pe.edit(
            table_name=TABLE_NAME,
            rowid=rowid,
            changes={"salary": new_salary},
            change_event_reason="REVIEW",
            change_comment=f"Oppdatering gjort av {user}",
        )
        elapsed = time.perf_counter() - t_start
        results.append((user, "OK", new_salary, elapsed))
        logger.info("%-8s  ✓  salary=%d  (%.3fs)", user, new_salary, elapsed)
    except Exception as e:
        elapsed = time.perf_counter() - t_start
        results.append((user, f"FEILET: {e}", None, elapsed))
        logger.warning("%-8s  ✗  %.3fs  %s", user, elapsed, e)
    finally:
        pe.close()


def print_summary(results: list, pe: ParquEdit) -> None:
    final = pe.view(TABLE_NAME, columns=["salary"])
    final_salary = int(final["salary"].iloc[0])

    ok = [r for r in results if r[1] == "OK"]
    failed = [r for r in results if r[1] != "OK"]
    times = [r[3] for r in results]

    print("\n" + "=" * 65)
    print("RESULTAT")
    print("=" * 65)
    for user, status, salary, elapsed in sorted(results):
        salary_str = f"  salary={salary}" if salary else ""
        print(f"  {user:<10}  {elapsed:.3f}s  {status}{salary_str}")
    print("-" * 65)
    print(f"  Vellykkede:       {len(ok)}")
    print(f"  Feilet:           {len(failed)}")
    print(f"\n  Raskeste:         {min(times):.3f}s")
    print(f"  Tregeste:         {max(times):.3f}s")
    print(f"  Gjennomsnitt:     {sum(times) / len(times):.3f}s")
    print(f"\n  Endelig lønn i tabell: {final_salary}")
    print("=" * 65)


def main() -> None:
    # Brukes kun til setup og oppsummering — ikke delt med trådene
    pe = ParquEdit()

    rowid = setup_table(pe)
    logger.info("Starter simulering med %d samtidige brukere...\n", len(USERS))

    results: list = []
    barrier = threading.Barrier(len(USERS))
    threads = [
        threading.Thread(
            target=simulate_user,
            args=(
                rowid,
                user,
                50000 + (i + 1) * 1000,
                results,
                barrier,
            ),
        )
        for i, user in enumerate(USERS)
    ]

    # Patch ssb_parquedit.dml.get_dapla_user globalt for hele simuleringen.
    # _simulated_get_dapla_user() er thread-safe via threading.local(),
    # så alle tråder får riktig bruker uten å interferere med hverandre.
    with patch.object(dml_module, "get_dapla_user", _simulated_get_dapla_user):
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    print_summary(results, pe)
    pe.close()


if __name__ == "__main__":
    main()