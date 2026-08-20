# WELFake label validation

The downloaded original WELFake CSV has 72,134 records: 35,028 with raw label 0 and 37,106 with raw label 1. The Zenodo record describes the corpus as 35,028 real and 37,106 fake articles. Therefore the released file is mapped as **raw 0 → real (project label 1)** and **raw 1 → fake (project label 0)**.

This decision is explicitly implemented in `src/data_loader.py`. The initial external results using the opposite mapping are invalidated and will not be used in reporting.
