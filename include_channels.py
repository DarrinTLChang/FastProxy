"""
Channel include list (allowlist) for `fastProxyV7.py`.

Edit this file to control which channels are processed.

Rules when `INCLUDE_ENABLE = True` in `fastProxyV7.py`:
- Only REGION_SIDE keys listed here will be processed.
- Only channels listed under each REGION_SIDE will be used.

Tip: comment out any channel line you don't want.
"""

INCLUDE_CHANNELS = {
    # ---- GPi1 Left ----
    "GPi1_L": [
        # 1,
        # # 2,
        # # 3,
        # 4,
        # 5,
        # # 6,
        # # 7,
        # 8,
        # 9,
        # 10,
    ],

    # ---- GPi1 Right ----
    "GPi1_R": [
        # 1,
        # 2,
        # 3,
        # 4,
        # 5,
        # 6,
        # 7,
        # 8,
        # 9,
        # 10,
    ],

    # ---- GPi2 Left ----
    "GPi2_L": [
        # 1,
        # 2,
        # 3,
        # 4,
        # 5,
        # 6,
        # 7,
        # 8,
        # 9,
        # 10,
    ],

    # ---- GPi2 Right ----
    "GPi2_R": [
        # 1,
        # 2,
        # 3,
        # 4,
        # 5,
        # 6,
        # 7,
        # 8,
        # 9,
        # 10,
    ],

    # ---- VO Left ----
    "Vo_L": [
        # 1,
        2,
        # 3,
        # 4,
        5,
        6,
        7,
        # 8,
        9,
        # 10,
    ],

    # ---- VO Right ----
    "Vo_R": [
        # 1,
        # 2,
        # 3,
        # 4,
        # 5,
        # 6,
        # 7,
        # 8,
        # 9,
        # 10,
    ],

    # ---- VIM Left ----
    "VIMPPN_L": [
        # 1,
        # 2,
        # 3,
        # 4,
        # # 5,
        # # 6,
        # 7,
        # 8,
        # # 9,
        # 10,
    ],

    # ---- VIM Right ----
    "VIMPPN_R": [
        # 1,
        # 2,
        # 3,
        # 4,
        # # 5,
        # # 6,
        # 7,
        # 8,
        # 9,
        # 10,
    ],

    # ---- VA Left ----
    "VA_L": [
        # 1,
        # 2,
        # 3,
        # # 4,
        # # 5,
        # 6,
        # 7,
        # 8,
        # 9,
        # 10,
    ],

    # ---- VA Left ----
    "VA_R": [
        # 1,
        # 2,
        # 3,
        # 4,
        # 5,
        # 6,
        # 7,
        # 8,
        # 9,
        # 10,
    ],

    # # ---- NA Left ----
    # "NA_L": [
    #     # 1,
    #     # 2,
    #     # 3,
    #     # 4,
    #     # 5,
    #     # 6,
    #     # 7,
    #     # 8,
    #     # 9,
    #     # 10,
    # ],

    # # ---- NA Right ----
    # "NA_R": [
    #     # 1,
    #     # 2,
    #     # 3,
    #     # 4,
    #     # 5,
    #     # 6,
    #     # 7,
    #     # 8,
    #     # 9,
    #     # 10,
    # ],
}

