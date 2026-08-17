"""nesgpt — nanoGPT-style training on the NES-MDB chiptune dataset.

Pipeline::

    python -m nesgpt.prepare        # TX1 files -> token .bin files
    python -m nesgpt.train          # train a small GPT
    python -m nesgpt.sample         # generate .tx1.txt songs
    python -m nesgpt.tx1_render     # TX1 -> WAV via chiptunify
"""

__version__ = "0.1.0"
